"""Throughput benchmark — the numbers behind docs/benchmarks.md, reproducible locally.

Run it: `uv run chad-bench` (downloads the same model `chad` uses on first run).

It drives the *real* `Engine` on the *real* model and reports the three numbers that
actually decide whether a local agent feels responsive:

  1. Prefill (cold)  — tok/s the model reads a fresh prompt at. The bill in a naive loop.
  2. Decode          — tok/s it writes new tokens at. Memory-bandwidth bound, ~constant.
  3. Warm step       — the agentic-loop number: how few tokens a *follow-up* turn has to
                       prefill once the persistent prefix cache is warm. This is the whole
                       point — a real turn pays ~0.1 s of prefill, not the cold-prefill time.

Everything here uses only public code and the public model, so the headline numbers in
the docs are something you can reproduce on your own Mac, not pass-rates from a private
eval suite.
"""
from __future__ import annotations

import argparse
import sys

# Reuse the exact model selection chad itself uses (RAM-aware default, local-dir-preferred,
# HF fallback) so the benchmark measures the model you'd actually run.
from .cli import _ensure_model, _pick_model
from .engine import Engine

# A chunk of plausible code/transcript text. We tile it to the requested token budget so
# the prefill measurement runs over realistic content (not a single repeated token, which
# the n-gram path could short-circuit). Decode speed is content-independent (bandwidth).
_FILLER = '''
def process(records, *, limit=None, strict=False):
    """Normalize and validate a batch of records, returning (ok, errors)."""
    ok, errors = [], []
    for i, rec in enumerate(records):
        if limit is not None and len(ok) >= limit:
            break
        try:
            name = rec["name"].strip()
            value = int(rec.get("value", 0))
        except (KeyError, ValueError, AttributeError) as exc:
            errors.append((i, str(exc)))
            if strict:
                raise
            continue
        ok.append({"name": name, "value": value, "index": i})
    return ok, errors
'''


def _build_prompt(tok, target_tokens: int) -> list:
    """Tile filler text and encode to roughly `target_tokens` token ids."""
    chunk_ids = tok.encode(_FILLER)
    if not chunk_ids:
        chunk_ids = tok.encode("the quick brown fox jumps over the lazy dog ")
    reps = max(1, target_tokens // len(chunk_ids) + 1)
    ids = (chunk_ids * reps)[:target_tokens]
    return ids


def _encode_suffix(tok, text: str) -> list:
    """Encode a short follow-up turn without a leading BOS (it's mid-sequence)."""
    try:
        return tok.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tok.encode(text)
        # Drop a leading BOS if the tokenizer added one.
        bos = getattr(tok, "bos_token_id", None)
        if bos is not None and ids and ids[0] == bos:
            ids = ids[1:]
        return ids


def _render(tok, messages, thinking=True) -> list:
    """Render a transcript exactly as Agent._render does (chat template + tool schemas
    + generation prompt), so the benchmark exercises the same token stream the real
    agentic loop feeds the engine."""
    from .tools import active_schemas
    return tok.apply_chat_template(
        messages, tools=active_schemas(), add_generation_prompt=True,
        enable_thinking=thinking)


def _agentic(model_id: str, why: str, context_tokens: int, apply_fix: bool):
    """Drive the REAL engine through a scripted agentic session that forces a
    truncated (token-cap) turn at large context, and measure the prefill the FOLLOWING
    step pays. This is the cache-miss the logs show: a turn cut off mid-`<think>` leaves
    the stored assistant turn unable to re-tokenize into a prefix of the non-trimmable
    KV cache, so the next step re-prefills the whole transcript (0 cached).

    With `apply_fix`, the dangling think block is closed before the turn is stored
    (chad.agent.close_unclosed_think) — the same normalization run_turn now applies — so
    the cached tokens stay a prefix and the next step prefills only a handful of tokens.

    Returns (miss_prefill_tokens, miss_prefill_s, cached_tokens)."""
    from .agent import build_system_prompt, close_unclosed_think
    from .engine import Engine

    eng = Engine(model_id=model_id, draft_id=None, cache_dir=None)
    eng.load()
    tok = eng.tok

    # Seed a large, realistic context: system + a user task that pastes a big code blob,
    # padded to roughly `context_tokens` so the re-prefill cost is at the scale the user
    # hit (tens of thousands of tokens).
    blob = (_FILLER * max(1, context_tokens // max(1, len(tok.encode(_FILLER)))))
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": "Review this module and fix the validation bug.\n\n"
                                     "```python\n" + blob + "\n```"},
    ]

    # The render up to the generation prompt: the template auto-opens `<think>\n` here, so
    # the model's turn continues INSIDE the think block.
    p1 = _render(tok, messages)
    # A turn that hit the token cap MID-THINK: reasoning with no closing `</think>` (the
    # `hit_cap=True` truncation the logs flag). Constructed deterministically rather than
    # generated, so the benchmark reliably exercises the divergence (a short stochastic
    # generation may close `</think>` within the cap and hide it).
    # Ends on punctuation (not bare whitespace), as a real cap-hit truncation does — a
    # token ending in a trailing space would merge under BPE on re-render and cost a
    # 1-token tail divergence that no text-level close can avoid on a non-trimmable cache.
    reasoning = (("Let me trace the validation path and reason about the boundary "
                  "conditions step by step. ") * 24).strip()
    gen_ids = tok.encode(reasoning, add_special_tokens=False)
    # Put the engine in the EXACT state generate() leaves after that truncated turn:
    # the live KV cache holds prompt + the raw (unclosed) reasoning tokens.
    cache_ids = list(p1) + gen_ids
    eng._reset_cache()
    eng._prefill(cache_ids)
    eng._cached_ids = cache_ids

    # Store the assistant turn the way run_turn does — with or without the fix — then
    # append a canned tool result (as the loop would after a tool call).
    stored = close_unclosed_think(reasoning, True) if apply_fix else reasoning
    messages.append({"role": "assistant", "content": stored})
    messages.append({"role": "tool", "name": "read",
                     "content": "def validate(x):\n    return x is not None\n"})

    # The FOLLOWING render. Its prefill is the cache miss we're measuring: with the fix
    # off the unclosed think makes the stored turn re-tokenize so it is no longer a prefix
    # of the cache -> non-trimmable reset -> full re-prefill. With the fix on the cache
    # stays warm and only the appended tokens prefill.
    p2 = _render(tok, messages)
    _, s2 = eng.generate(p2, max_tokens=8)
    return s2.prompt_tokens, s2.prefill_s, s2.cached_tokens


def _run_agentic(model_id: str, why: str, context_tokens: int) -> int:
    w = 72
    print(f"\nloading {model_id} [{why}] ...", flush=True)
    # Fresh process-cold cache for each variant (separate Engine in _agentic).
    off_new, off_s, off_cached = _agentic(model_id, why, context_tokens, apply_fix=False)
    on_new, on_s, on_cached = _agentic(model_id, why, context_tokens, apply_fix=True)
    ctx = on_new + on_cached  # actual rendered transcript size at the measured step
    print("=" * w)
    print(f"agentic prefill — truncated-turn cache miss @ {ctx:,} ctx tokens")
    print("=" * w)
    print(f"{'':22}{'new prefill':>14}{'cached':>10}{'prefill s':>12}")
    print("-" * w)
    print(f"{'truncation, fix OFF':22}{off_new:>14,}{off_cached:>10,}{off_s:>12.2f}")
    print(f"{'truncation, fix ON':22}{on_new:>14,}{on_cached:>10,}{on_s:>12.2f}")
    print("-" * w)
    saved = off_new - on_new
    speedup = (off_s / on_s) if on_s else float("inf")
    print(f"fix avoids re-prefilling {saved:,} tokens on the step after a truncation "
          f"(~{off_s - on_s:.1f}s, {speedup:.0f}x faster prefill).")
    print("=" * w + "\n")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="chad-bench",
        description="Measure chad's prefill / decode / warm-step throughput on this machine.",
    )
    ap.add_argument("--prefill-tokens", type=int, default=5000,
                    help="size of the cold prompt to prefill (default: 5000)")
    ap.add_argument("--gen-tokens", type=int, default=128,
                    help="tokens to decode for the decode measurement (default: 128)")
    ap.add_argument("--agentic", action="store_true",
                    help="run the agentic-session prefill benchmark: reproduce the "
                         "truncated-turn cache miss and the fix, measuring the re-prefill "
                         "the next step pays (with/without the think-close fix)")
    ap.add_argument("--context-tokens", type=int, default=24000,
                    help="seeded context size for --agentic (default: 24000)")
    args = ap.parse_args(argv)

    if args.agentic:
        model_id, why = _pick_model()
        _ensure_model(model_id)
        return _run_agentic(model_id, why, args.context_tokens)

    model_id, why = _pick_model()
    _ensure_model(model_id)

    # cache_dir=None so the on-disk warm-prefix can't pre-load and skew the *cold* prefill
    # number; this benchmark measures from a genuinely cold cache.
    eng = Engine(model_id=model_id, draft_id=None, cache_dir=None)
    sys.stderr.write(f"loading {model_id} [{why}] ...\n")
    load_s = eng.load()

    prompt_ids = _build_prompt(eng.tok, args.prefill_tokens)

    # 1 + 2: cold prefill, then decode. One generate() call does both; GenStats splits the
    # time into prefill_s (reading the prompt) and gen_s (writing new tokens).
    _, stats = eng.generate(prompt_ids, max_tokens=args.gen_tokens)
    prefill_tps = stats.prompt_tokens / stats.prefill_s if stats.prefill_s else 0.0
    decode_tps = stats.tok_per_s

    # 3: warm step. The cache now holds the whole prompt + what we just generated. A real
    # agentic turn appends a small user/tool message and regenerates — a strict extension,
    # so only the appended tokens prefill. Build that follow-up from the live cached ids.
    suffix = _encode_suffix(eng.tok, "\n\nNow add a docstring to process() and rerun the tests.\n")
    warm_prompt = list(eng._cached_ids) + suffix
    _, wstats = eng.generate(warm_prompt, max_tokens=8)

    w = 64
    print("\n" + "=" * w)
    print(f"chad throughput — {model_id}")
    print(f"  {why}")
    print("=" * w)
    print(f"model load               {load_s:6.1f} s")
    print("-" * w)
    print(f"1. prefill (cold)        {stats.prompt_tokens:6d} tok in {stats.prefill_s:6.2f} s"
          f"   ->  {prefill_tps:6.0f} tok/s")
    print(f"2. decode                {stats.generated_tokens:6d} tok in {stats.gen_s:6.2f} s"
          f"   ->  {decode_tps:6.1f} tok/s")
    print(f"3. warm step             {wstats.prompt_tokens:6d} new tok prefilled "
          f"({wstats.cached_tokens} cached)")
    print(f"                         {wstats.prefill_s:6.2f} s of prefill for the follow-up turn")
    print("-" * w)
    cold_est = (stats.prompt_tokens / prefill_tps) if prefill_tps else 0.0
    print(f"the agentic-loop win: a follow-up turn prefills {wstats.prompt_tokens} tokens "
          f"(~{wstats.prefill_s:.2f} s),")
    print(f"not the full {wstats.cached_tokens + wstats.prompt_tokens} "
          f"(~{cold_est:.1f} s) a cache-less backend would re-read every step.")
    print("=" * w + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
