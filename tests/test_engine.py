"""Regression tests for the engine's prompt-lookup decoding (PLD) and its prefix-cache
reuse contract — the most correctness-critical code in the repo.

Tier 1 (always runs, no model): pure-function characterization of
`engine.prompt_lookup_draft`, the numpy n-gram drafter. Every expected value is worked
out by hand from the live docstring (longest match wins, ties broken by recency, copy
what followed). Contexts are kept tiny so a reviewer can re-derive each one.

Tier 2 (model-gated, self-skipping): loads a small *trimmable, draft-less* model
(Qwen2.5-Coder-0.5B-Instruct-4bit — NOT Ornith, which is non-trimmable so PLD never
engages) and asserts PLD produces output byte-identical to plain greedy decoding. The
engine returns decoded text (not raw ids), so equivalence is asserted on the full
decoded output string — the user-observable signal a corruption bug would break — plus a
check that speculation actually engaged (`forwards < generated_tokens`). Follows the
self-skip idiom from test_localization.py: if the model can't load or the cache isn't
trimmable, it prints SKIP and continues — it never fails for a missing model.

If tier 2 runs and finds PLD output != greedy output, that is a real corruption bug:
the assert is intentionally strict — do NOT loosen it.

RESOLVED (plan 010): the stale `scratchpad/pld_probe.py` reference in engine.py (it cited
a deleted file as the bit-exact check for the *hybrid* recurrent-snapshot rollback) has
been dropped — engine.py's comment now points the *trimmable* PLD equivalence at this
file (tier 2) and notes the hybrid rollback has no automated case yet (no hybrid model is
loaded in the test env). Plan 010 also added tier-2b cache-reuse correctness cases below.

Run: `.venv/bin/python test_engine.py`
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chad.engine import prompt_lookup_draft  # noqa: E402

PASS = 0
FAIL = 0
SKIP = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        raise AssertionError(f"{name}  {detail}")


def skip(name, why):
    global SKIP
    SKIP += 1
    print(f"  SKIP: {name}  ({why})")
    if os.environ.get("PYTEST_CURRENT_TEST"):
        import pytest
        pytest.skip(why)


def _is_quantized(model_dir):
    """True if the model dir's config.json declares quantization (mlx/awq/gptq).

    Bit-exact greedy equivalence is undefined on a quantized hybrid-SSM model: the coarse
    logit grid produces exact argmax ties (top1==top2) where the winner is decided by
    sub-ULP FP noise from how tokens are batched into forward passes — so even two
    legitimate greedy decodes of the same model can disagree at a tie. (Verified 2026-06:
    the strict test below passes on bf16 Ornith-9B and on quantized trimmable Qwen, but
    'fails' on 4-bit AWQ Ornith purely at margin-0.0 ties — not a rollback bug.) So when
    the env var points at quantized weights we SKIP rather than fail spuriously; the
    corruption guard stays strict for the bf16 run where it is meaningful."""
    import json
    cfg = os.path.join(model_dir, "config.json")
    try:
        with open(cfg) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return False  # can't read config → don't skip; let the strict path run
    return "quantization" in data or "quantization_config" in data


# === Tier 1: pure-function tests for prompt_lookup_draft ======================
# Live semantics (engine.py:54): try ngram_max..ngram_min, pattern = the current
# suffix arr[-ng:]; among earlier occurrences (starts in arr[:n-ng]), take the
# rightmost (most recent) full match; return up to num_draft tokens that followed it.
# Guards: [] when len(context) < 2 or num_draft <= 0; [] on no match.

def test_longest_match_wins():
    # Both a 3-gram (1,2,3 -> 9) and a 1-gram (3 -> 9 and 3 -> end) could match the
    # suffix's tail. Longest wins: the earlier "1,2,3" at index 0 is followed by 9.
    ctx = [1, 2, 3, 9, 4, 5, 1, 2, 3]
    check("longest-match-wins returns 3-gram continuation",
          prompt_lookup_draft(ctx, num_draft=1, ngram_max=3) == [9],
          repr(prompt_lookup_draft(ctx, num_draft=1, ngram_max=3)))


def test_recency_tie_break():
    # Suffix [7,8] occurs earlier at index 0 (followed by 5) and index 3 (followed by
    # 6). Recency tie-break -> the rightmost earlier occurrence (index 3) wins -> [6].
    ctx = [7, 8, 5, 7, 8, 6, 7, 8]
    got = prompt_lookup_draft(ctx, num_draft=1, ngram_max=2)
    check("recency tie-break picks rightmost continuation", got == [6], repr(got))


def test_no_match():
    # No suffix of [..,3,4,5] occurs earlier in the context -> [].
    ctx = [1, 2, 3, 4, 5]
    got = prompt_lookup_draft(ctx, num_draft=1, ngram_max=3)
    check("no-match returns empty", got == [], repr(got))


def test_guards():
    # num_draft <= 0 -> [] even with an obvious match.
    check("num_draft=0 guard", prompt_lookup_draft([1, 2, 1, 2], 0) == [])
    check("num_draft<0 guard", prompt_lookup_draft([1, 2, 1, 2], -3) == [])
    # len(context) < 2 -> [].
    check("single-element context guard", prompt_lookup_draft([1], 3) == [])
    check("empty context guard", prompt_lookup_draft([], 3) == [])


def test_draft_length():
    # Long matching run: earlier "1,2,3" at index 0 is followed by 4,5,6 -> full 3.
    ctx = [1, 2, 3, 4, 5, 6, 9, 1, 2, 3]
    got = prompt_lookup_draft(ctx, num_draft=3, ngram_max=3)
    check("draft length returns up to num_draft", got == [4, 5, 6], repr(got))
    # Match near the end of context yields FEWER than num_draft: in [1,2,1,2] the
    # earlier [1,2] at index 0 is followed by only context[2:5] = [1,2] (2 tokens).
    ctx2 = [1, 2, 1, 2]
    got2 = prompt_lookup_draft(ctx2, num_draft=3, ngram_max=3)
    check("draft truncated near end of context", got2 == [1, 2], repr(got2))


# === Tier 2: PLD == plain-greedy equivalence (model-gated, self-skipping) =====

# A small dense Qwen2.5-Coder is trimmable; Ornith is not (so PLD never engages).
TIER2_MODEL = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"


def _build_engine(model_id=TIER2_MODEL, enable_pld_hybrid=False):
    """Load a small draft-less PLD engine, or return None to SKIP.

    `enable_pld_hybrid` opts into the hybrid (qwen3_5/Ornith) recurrent-snapshot
    rollback path (off by default; see Engine.enable_pld_hybrid).
    """
    try:
        from chad.engine import Engine
        eng = Engine(
            model_id=model_id,
            draft_id=None,            # draft-less: PLD only engages when draft is None
            prompt_lookup=True,
            temp=0.0,
            kv_bits=None,
            enable_pld_hybrid=enable_pld_hybrid,
        )
        eng.load()
        return eng
    except Exception as e:  # noqa: BLE001 — missing model / MLX issue -> self-skip
        print(f"  (tier 2 load failed: {type(e).__name__}: {e})")
        return None


def test_pld_equals_greedy():
    # The aggregate runner (run_tests.py) sets CHAD_FAST_TESTS=1 for its fast default
    # gate; this tier loads a real model (seconds–minutes when cached) so skip it there.
    # It still runs under `run_tests.py --evals` (which does NOT set the var).
    if os.environ.get("CHAD_FAST_TESTS"):
        return skip("pld_equals_greedy", "CHAD_FAST_TESTS set (fast gate; skips model load)")
    eng = _build_engine()
    if eng is None:
        return skip("pld_equals_greedy", f"could not load {TIER2_MODEL}")
    if not eng._trimmable:
        return skip("pld_equals_greedy", "cache not trimmable; PLD path won't engage")

    # Quote-heavy prompt: ask the model to repeat a chunk verbatim, so generation
    # re-emits context tokens — PLD's best case (the n-gram lookup nails long runs).
    snippet = (
        "def fibonacci(n):\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        a, b = b, a + b\n"
        "    return a\n"
    )
    messages = [
        {"role": "system", "content": "You are a precise coding assistant."},
        {"role": "user", "content":
            "Repeat the following Python function verbatim inside a code block, then "
            "write one sentence describing what it returns.\n\n" + snippet},
    ]
    prompt_ids = eng.tok.apply_chat_template(messages, add_generation_prompt=True)
    max_tokens = 180

    # PLD run.
    eng.prompt_lookup = True
    eng._reset_cache()
    pld_text, pld_stats = eng.generate(list(prompt_ids), max_tokens=max_tokens)

    # Plain greedy run (PLD disabled) — same temp=0 argmax decode.
    eng.prompt_lookup = False
    eng._reset_cache()
    greedy_text, greedy_stats = eng.generate(list(prompt_ids), max_tokens=max_tokens)

    # CORRUPTION GUARD: identical observable output. If this fails it is a real
    # accept/rollback bug — do NOT loosen the assert; STOP and report (plan 005).
    check("PLD output == plain greedy output", pld_text == greedy_text,
          f"\n--- PLD ---\n{pld_text!r}\n--- GREEDY ---\n{greedy_text!r}")

    # Sanity: PLD must have actually USED speculation (fewer forwards than tokens),
    # else the corruption guard above is vacuous — greedy==greedy with PLD doing
    # nothing trivially passes and the accept/rollback path it exists to protect is
    # never exercised. This prompt is built to repeat (it quotes its own context), so
    # a no-accept run means PLD is broken, not that the test is wrong: FAIL, not WARN.
    check("PLD speculation engaged (forwards < generated_tokens)",
          pld_stats.forwards < pld_stats.generated_tokens,
          f"PLD accepted no speculation (forwards={pld_stats.forwards} >= "
          f"generated={pld_stats.generated_tokens}); accept path not exercised — "
          f"corruption guard above is vacuous")


# === Tier 2-hybrid: hybrid-PLD == plain-greedy equivalence (real hybrid weights, self-skipping) ===
# The trimmable PLD path above is pinned bit-exact. This is its analogue for the
# HIGHEST-corruption-risk code in the repo: the hybrid (qwen3_5/Ornith) recurrent-
# snapshot rollback (engine.py: _snap_recurrent -> _restore_recurrent -> _trim_kv ->
# re-feed the accepted prefix). On a rejected speculative draft that sequence must land
# the recurrent DeltaNet state and the attention KV at EXACTLY y_val+draft[:n_acc]; one
# token off and every subsequent turn silently generates wrong tokens.
#
# A hybrid (non-trimmable) model is required to exercise this; CI runners are weightless,
# so the test reads the model dir from CHAD_TEST_HYBRID_MODEL and self-skips when unset.
# It must NOT hardcode a path (the local Ornith weights live in the private workshop, not
# the public repo). The bit-equality guard requires UNQUANTIZED (bf16) weights — on a
# quantized hybrid model the coarse logit grid yields exact argmax ties that bit-equality
# can't survive, so the test self-skips on quantized dirs (see _is_quantized). To run it:
#   CHAD_TEST_HYBRID_MODEL=<path-to-bf16-ornith-dir> uv run pytest \
#       tests/test_engine.py::test_pld_hybrid_equals_greedy -q

def test_pld_hybrid_equals_greedy():
    # Fast gate skips model loads (matches test_pld_equals_greedy).
    if os.environ.get("CHAD_FAST_TESTS"):
        return skip("pld_hybrid_equals_greedy",
                    "CHAD_FAST_TESTS set (fast gate; skips model load)")
    model_id = os.environ.get("CHAD_TEST_HYBRID_MODEL")
    if not model_id:
        return skip("pld_hybrid_equals_greedy",
                    "CHAD_TEST_HYBRID_MODEL unset (no hybrid weights in test env)")
    # Quantized weights make bit-exact greedy equivalence undefined (margin-0.0 argmax
    # ties decided by FP batching noise — see _is_quantized). Point this at the bf16
    # source weights (e.g. private/models-src/Ornith-1.0-9B) to exercise the guard.
    if _is_quantized(model_id):
        return skip("pld_hybrid_equals_greedy",
                    f"{model_id} is quantized; bit-equality is undefined under "
                    "quantization exact-ties — run against bf16 weights")
    eng = _build_engine(model_id=model_id, enable_pld_hybrid=True)
    if eng is None:
        return skip("pld_hybrid_equals_greedy", f"could not load {model_id}")
    # Guard against pointing the env var at a trimmable model by mistake: the hybrid
    # rollback path only engages on a non-trimmable, pld_hybrid cache.
    if eng._trimmable or not eng._pld_hybrid:
        return skip("pld_hybrid_equals_greedy",
                    "not a hybrid (non-trimmable, pld_hybrid) cache; rollback path won't engage")

    # Same quote-heavy prompt as test_pld_equals_greedy: ask the model to repeat a chunk
    # verbatim so generation re-emits context tokens and the n-gram drafter accepts long
    # runs — that drives partial rejections through the recurrent-snapshot rollback.
    snippet = (
        "def fibonacci(n):\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        a, b = b, a + b\n"
        "    return a\n"
    )
    messages = [
        {"role": "system", "content": "You are a precise coding assistant."},
        {"role": "user", "content":
            "Repeat the following Python function verbatim inside a code block, then "
            "write one sentence describing what it returns.\n\n" + snippet},
    ]
    prompt_ids = eng.tok.apply_chat_template(messages, add_generation_prompt=True)
    max_tokens = 180

    # Hybrid-PLD run (recurrent-snapshot rollback active).
    eng.prompt_lookup = True
    eng.enable_pld_hybrid = True
    eng._reset_cache()
    pld_text, pld_stats = eng.generate(list(prompt_ids), max_tokens=max_tokens)

    # Plain greedy run (PLD disabled) — same temp=0 argmax decode.
    eng.prompt_lookup = False
    eng._reset_cache()
    greedy_text, greedy_stats = eng.generate(list(prompt_ids), max_tokens=max_tokens)

    # CORRUPTION GUARD: identical observable output. If this fails it is a real
    # recurrent-rollback bug — do NOT loosen the assert; STOP and report (plan 030).
    check("hybrid PLD output == plain greedy output", pld_text == greedy_text,
          f"\n--- HYBRID PLD ---\n{pld_text!r}\n--- GREEDY ---\n{greedy_text!r}")

    # Sanity: hybrid PLD must have actually USED speculation (fewer forwards than
    # tokens), else the corruption guard is vacuous — the snapshot/restore/trim/re-feed
    # path it exists to protect never ran. The quote-heavy prompt is built to accept, so
    # a no-accept run means PLD is broken, not that the test is wrong: FAIL, not WARN.
    check("hybrid PLD speculation engaged (forwards < generated_tokens)",
          pld_stats.forwards < pld_stats.generated_tokens,
          f"hybrid PLD accepted no speculation (forwards={pld_stats.forwards} >= "
          f"generated={pld_stats.generated_tokens}); rollback path not exercised — "
          f"corruption guard above is vacuous")


# === Tier 2b: cache-reuse correctness on the truncation/degenerate paths (plan 010) ==
# These force the two rare cache-fast-path branches that plan 010 fixed and assert
# byte/text identity to the no-splice / fresh-cache baseline. The optimization must
# NEVER change output; if either fails, the cache edit is wrong — do NOT loosen.

def test_degenerate_reprefill_matches_fresh():
    """Site B: regenerating an identical prompt forces generate()'s degenerate
    `not suffix` branch (prompt fully cached). With the off-by-one fixed, the live KV
    cache is kept in lockstep with _cached_ids, so the second (degenerate-path)
    generation must be byte-identical to a fresh-cache greedy run."""
    if os.environ.get("CHAD_FAST_TESTS"):
        return skip("degenerate_reprefill", "CHAD_FAST_TESTS set (fast gate)")
    eng = _build_engine()
    if eng is None:
        return skip("degenerate_reprefill", f"could not load {TIER2_MODEL}")
    if not eng._trimmable:
        return skip("degenerate_reprefill", "cache not trimmable")

    messages = [
        {"role": "system", "content": "You are a precise coding assistant."},
        {"role": "user", "content": "Write a one-line Python function that squares n."},
    ]
    prompt_ids = list(eng.tok.apply_chat_template(messages, add_generation_prompt=True))
    max_tokens = 48

    # The degenerate branch lives in the standard (non-PLD) generate() path, so force
    # that path by disabling prompt-lookup (still greedy, temp=0).
    eng.prompt_lookup = False

    # Baseline: fresh-cache greedy run.
    eng._reset_cache()
    base_text, _ = eng.generate(list(prompt_ids), max_tokens=max_tokens)

    # Drive the degenerate branch: prime the cache with this prompt, then regenerate
    # the *identical* prompt so suffix becomes empty (`not suffix`).
    eng._reset_cache()
    eng.generate(list(prompt_ids), max_tokens=max_tokens)            # primes the cache
    degen_text, degen_stats = eng.generate(list(prompt_ids), max_tokens=max_tokens)

    check("degenerate re-prefill == fresh greedy (Site B, no off-by-one)",
          degen_text == base_text,
          f"\n--- DEGEN ---\n{degen_text!r}\n--- FRESH ---\n{base_text!r}")
    # Confirm we actually hit the degenerate branch (1 token re-fed, rest cached).
    check("degenerate branch fired (prompt_tokens == 1)",
          degen_stats.prompt_tokens == 1,
          f"prompt_tokens={degen_stats.prompt_tokens} (expected 1)")


def test_truncation_recovery_matches_fresh():
    """Site A: a turn that hits max_tokens is TRUNCATED — the engine caches its raw
    generated ids, which lack the closing `<|im_end|>\\n` the template appends when the
    text is re-rendered as a message. So the next step's full re-render DIVERGES from the
    warm cache partway through the assistant turn, and `engine._sync_to` must re-prefill
    from the divergence point. The recovery is correct iff generating from that warm,
    divergent cache is byte-identical to generating the same prompt from a FRESH cache —
    the cache reuse must never change output. (The old Agent._render_for_cache splice was
    proven to always equal a plain render, so it was deleted; the real invariant lives at
    the engine level and is what this test now pins.)"""
    if os.environ.get("CHAD_FAST_TESTS"):
        return skip("truncation_recovery", "CHAD_FAST_TESTS set (fast gate)")
    eng = _build_engine()
    if eng is None:
        return skip("truncation_recovery", f"could not load {TIER2_MODEL}")
    if not eng._trimmable:
        return skip("truncation_recovery", "cache not trimmable")
    from chad.agent import Agent

    eng.prompt_lookup = True
    agent = Agent(eng, mode="auto", thinking=False)
    agent.messages.append({"role": "user", "content": "Print hello world in Python."})
    base_prompt = list(agent._render())

    # Drive a real TRUNCATED turn: a tiny max_tokens leaves the cache holding
    # base_prompt + raw generated ids (no terminator).
    eng._reset_cache()
    trunc_text, _ = eng.generate(list(base_prompt), max_tokens=6)
    agent.messages.append({"role": "assistant", "content": trunc_text})
    agent.messages.append({"role": "user", "content": "now add a docstring"})

    # The recovery prompt diverges from the warm cache inside the assistant turn, so
    # the engine must reuse the common prefix and re-prefill the rest.
    recover = list(agent._render())
    warm_text, warm_stats = eng.generate(list(recover), max_tokens=24)

    # Same prompt from a fresh cache — the no-reuse baseline.
    eng._reset_cache()
    fresh_text, _ = eng.generate(list(recover), max_tokens=24)

    check("truncation recovery: warm divergent-cache gen == fresh gen (engine _sync_to)",
          warm_text == fresh_text,
          f"\n--- WARM ---\n{warm_text!r}\n--- FRESH ---\n{fresh_text!r}")
    # Prove the warm run actually REUSED cache (didn't silently re-prefill everything):
    # some prefix of `recover` must have been served from the truncated-turn cache.
    check("truncation recovery: warm run reused cached prefix (real divergence path)",
          warm_stats.cached_tokens > 0,
          f"cached_tokens={warm_stats.cached_tokens} (expected > 0 — divergence not exercised)")


if __name__ == "__main__":
    tier1 = (test_longest_match_wins, test_recency_tie_break, test_no_match,
             test_guards, test_draft_length)
    tier2 = (test_pld_equals_greedy,
             test_pld_hybrid_equals_greedy,
             test_degenerate_reprefill_matches_fresh,
             test_truncation_recovery_matches_fresh)
    for fn in tier1 + tier2:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            FAIL += 1
            print(f"  ERROR in {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{PASS} passed, {FAIL} failed, {SKIP} skipped")
    raise SystemExit(1 if FAIL else 0)
