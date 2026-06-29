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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="chad-bench",
        description="Measure chad's prefill / decode / warm-step throughput on this machine.",
    )
    ap.add_argument("--prefill-tokens", type=int, default=5000,
                    help="size of the cold prompt to prefill (default: 5000)")
    ap.add_argument("--gen-tokens", type=int, default=128,
                    help="tokens to decode for the decode measurement (default: 128)")
    args = ap.parse_args(argv)

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
