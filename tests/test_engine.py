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

RESOLVED: the stale `scratchpad/pld_probe.py` reference in engine.py (it cited
a deleted file as the bit-exact check for the *hybrid* recurrent-snapshot rollback) has
been dropped — engine.py's comment now points the *trimmable* PLD equivalence at this
file (tier 2) and notes the hybrid rollback has no automated case yet (no hybrid model is
loaded in the test env). Tier-2b cache-reuse correctness cases follow below.

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


def test_prefill_progress_callback():
    """Tier 1 (no weights): `_prefill` reports per-chunk progress via on_progress(done,
    total) — monotonic, ending exactly at total — and runs identically with NO callback.

    The added line is `if on_progress: on_progress(i, n)` AFTER the model forward + i
    increment, so with no callback the cache-feeding loop is byte-identical. We prove
    that here by asserting the sequence of forward-pass shapes is the same with and
    without the callback (full decoded-output equivalence is the model-gated tier-2
    job; this path feeds the non-trimmable cache, so the no-op guarantee matters)."""
    import mlx.core as mx

    from chad.engine import Engine

    class _FakeCacheItem:
        def __init__(self):
            self.state = mx.array([0.0])

    def _make_engine(forward_shapes):
        eng = object.__new__(Engine)  # bypass __init__ (no weights to load)
        eng._cache = [_FakeCacheItem(), _FakeCacheItem()]
        eng.model = lambda arr, cache=None: forward_shapes.append(int(arr.shape[1]))
        return eng

    # 70 tokens, chunk 32 -> forwards of 32,32,6; progress fires done=32,64,70.
    shapes, seen = [], []
    fed = _make_engine(shapes)._prefill(
        list(range(70)), chunk=32, on_progress=lambda d, t: seen.append((d, t)))
    dones = [d for d, _ in seen]
    check("prefill returns full count", fed == 70, f"fed={fed}")
    check("progress fired per chunk, ending at total", dones == [32, 64, 70], str(seen))
    check("progress total constant", all(t == 70 for _, t in seen), str(seen))
    check("progress monotonic", dones == sorted(dones), str(dones))

    # No callback: same forward-pass shapes, same fed count -> instrumentation is a no-op.
    shapes_none = []
    fed_none = _make_engine(shapes_none)._prefill(list(range(70)), chunk=32)
    check("no-callback fed identical", fed_none == fed, f"{fed_none} vs {fed}")
    check("no-callback forward shapes identical", shapes_none == shapes,
          f"{shapes_none} vs {shapes}")


def test_interrupted_prefill_records_only_fed_tokens():
    """Tier 1 (no weights): a mid-prefill interrupt must leave `_cached_ids` recording
    ONLY the tokens actually pushed into the KV cache — never the full intended prefix.

    This is the append-only-cache invariant the design calls catastrophic if broken. The
    cache holds exactly `fed` tokens after an interrupt, so `_cached_ids` (the engine's
    record of what is resident) must have length `fed`. If it recorded the full prefix
    while the cache held fewer, every later turn's prefix-diff would count phantom
    resident tokens and splice the next prefill at the wrong offset — silent corruption of
    every subsequent turn. We drive the REAL cold-miss path in `warm_prefix` (engine.py
    ~294-297, which owns the `_cached_ids = prefix_ids[:fed]` accounting) with a
    `should_stop` that fires after the first chunk, then prove a clean resume of the
    remaining suffix rebuilds the full prefix with every token fed exactly once (no gap,
    no double-feed). Removing the `fed < len(prefix_ids)` guard (recording the full prefix)
    makes the length assertion below go RED."""
    import mlx.core as mx

    from chad.engine import Engine

    class _FakeCacheItem:
        def __init__(self):
            self.state = mx.array([0.0])

    forwards = []                          # width (in tokens) of every model forward pass
    eng = object.__new__(Engine)           # bypass __init__ (no weights)
    eng._cache = [_FakeCacheItem(), _FakeCacheItem()]
    eng.model = lambda arr, cache=None: forwards.append(int(arr.shape[1]))
    eng.draft = None                       # single-model cache -> warm_prefix is valid
    eng.cache_dir = "/tmp/chad-test-nonexistent"   # truthy so warm_prefix doesn't 'skip'
    eng._cached_ids = []                            # cold: nothing resident yet
    eng._reset_cache = lambda: None                 # keep our fake _cache in place
    eng._ckpt_path = lambda ids, tag=None: "/tmp/chad-test-nonexistent/no.ckpt"  # -> miss

    prefix = list(range(600))              # > 1 chunk at the 512-token default
    checks = {"n": 0}

    def should_stop():
        # False on the first check (feed chunk 0..512), True after (break before chunk 1).
        checks["n"] += 1
        return checks["n"] > 1

    status, fed = eng.warm_prefix(prefix, should_stop=should_stop)

    check("cold-miss path taken", status == "miss", status)
    check("prefill was interrupted (fed < full prefix)", fed < len(prefix), fed)
    check("interrupt fed exactly one 512-chunk", fed == 512, fed)
    # THE invariant: _cached_ids length == tokens actually pushed == sum of forward widths.
    check("_cached_ids records only the fed tokens (NOT the full prefix)",
          len(eng._cached_ids) == fed, f"len={len(eng._cached_ids)} vs fed={fed}")
    check("_cached_ids is the fed PREFIX exactly", eng._cached_ids == prefix[:fed],
          eng._cached_ids[:5])
    check("cache holds exactly fed tokens (forwards sum == _cached_ids)",
          sum(forwards) == len(eng._cached_ids),
          f"{sum(forwards)} vs {len(eng._cached_ids)}")

    # Clean resume: feed the remaining suffix into the SAME cache and extend _cached_ids as
    # the engine's diff/extend math would. Because the interrupt recorded only `fed`, the
    # resume starts at the correct offset — every token lands exactly once, no gap/overlap.
    fed2 = eng._prefill(prefix[fed:])          # no should_stop -> runs to completion
    eng._cached_ids = eng._cached_ids + prefix[fed:]
    check("resume fed exactly the remaining tail", fed2 == len(prefix) - fed, fed2)
    check("resumed cache == full prefix, once each", eng._cached_ids == prefix,
          f"len={len(eng._cached_ids)}")
    check("no token fed twice or skipped (total forwards == prefix length)",
          sum(forwards) == len(prefix), f"{sum(forwards)} vs {len(prefix)}")


def test_stop_condition_soft_close():
    """Tier 1 (no weights): generate()'s soft think-cap hook. A caller
    `stop_condition(text, n)` is checked after each decoded token, composing with
    should_stop; when it fires, generation halts mid-stream and stats.stop_condition_fired
    is set so run_turn can tell a deliberate early stop from an EOS/max_tokens finish.
    We drive it with a faked `stream_generate` (the model itself is never called on the
    decode path once tokens stream), asserting: the cap stops us while still inside
    <think> (before the faked </think>), the flag is set, and — critically — a None
    stop_condition is a byte-identical no-op (same tokens consumed, flag stays False)."""
    from chad import engine as eng_mod
    from chad.engine import Engine

    class _Resp:
        def __init__(self, text, token):
            self.text, self.token = text, token

    # Faked stream: 8 in-think chunks, then a </think>, then an answer chunk. A cap of 5
    # (still-in-think) must cut generation before the </think> token is ever reached.
    def _fake_stream(model, tok, arr, **kw):
        for i in range(8):
            yield _Resp("x", 1000 + i)
        yield _Resp("</think>", 2000)
        yield _Resp("ans", 2001)

    def _mk_engine():
        eng = object.__new__(Engine)  # bypass __init__ (no weights)
        eng.draft = None
        eng.prompt_lookup = False      # force the standard (non-PLD) generate path
        eng.temp = 0.0
        eng.kv_bits = None
        eng._trimmable = False
        eng._pld_hybrid = False
        eng.enable_pld_hybrid = False
        eng._cached_ids = []
        eng._sync_to = lambda ids: 0            # nothing cached -> whole prompt is suffix
        eng._prefill = lambda ids, ss=None, chunk=256, on_progress=None: len(ids)
        eng.model = object()
        eng.tok = object()
        return eng

    orig = eng_mod.stream_generate
    try:
        eng_mod.stream_generate = _fake_stream
        cap = 5

        def _sc(text_so_far, n):
            return n >= cap and "</think>" not in text_so_far

        eng = _mk_engine()
        text, stats = eng.generate([1, 2, 3, 4], max_tokens=100, stop_condition=_sc)
        check("stopped while still inside <think>", "</think>" not in text, repr(text))
        check("stop_condition_fired flag set", stats.stop_condition_fired is True)
        check("halted at the cap (not the full stream)",
              stats.generated_tokens == cap, stats.generated_tokens)

        # None stop_condition: byte-identical no-op. The faked stream self-terminates at
        # 10 tokens (max_tokens is higher), so all 10 are consumed and the flag stays off.
        eng2 = _mk_engine()
        text2, stats2 = eng2.generate([1, 2, 3, 4], max_tokens=100, stop_condition=None)
        check("no stop_condition consumes the whole stream",
              stats2.generated_tokens == 10, stats2.generated_tokens)
        check("no stop_condition leaves flag False",
              stats2.stop_condition_fired is False)
        check("no stop_condition emits the closed think + answer",
              "</think>" in text2 and text2.endswith("ans"), repr(text2))
    finally:
        eng_mod.stream_generate = orig


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
    # CI's fast gate (.github/workflows/tests.yml) sets CHAD_FAST_TESTS=1 to stay
    # model-free; this tier loads a real model (seconds–minutes when cached) so skip it
    # there. Run it locally by invoking pytest WITHOUT that var (weights on disk).
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
    # accept/rollback bug — do NOT loosen the assert; STOP and report.
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
    # source weights (e.g. a local bf16 Ornith-1.0-9B checkout) to exercise the guard.
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
    # recurrent-rollback bug — do NOT loosen the assert; STOP and report.
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


def test_hybrid_rewind_matches_fresh():
    """Tier 2 (hybrid weights): the bounded rewind must be INVISIBLE in
    output. Drive the exact production shape: turn 1 generates (leaving cache =
    prompt + generated ids and a turn-boundary snapshot), then the next prompt
    DIVERGES inside that generation (a truncated turn re-rendered — Site A on a
    hybrid). `_sync_to` must take the rewind path (cached_tokens > 0 through the
    divergence point, no full rebuild), and the resulting generation must equal the
    same prompt from a fresh engine. Runs strict bit-equality on bf16 weights; on
    quantized weights argmax exact-ties make bit-equality undefined (see
    _is_quantized), so it degrades to structural checks + a warm/fresh prefix
    sanity comparison."""
    if os.environ.get("CHAD_FAST_TESTS"):
        return skip("hybrid_rewind", "CHAD_FAST_TESTS set (fast gate)")
    model_id = os.environ.get("CHAD_TEST_HYBRID_MODEL")
    if not model_id:
        return skip("hybrid_rewind", "CHAD_TEST_HYBRID_MODEL unset")
    eng = _build_engine(model_id=model_id)
    if eng is None:
        return skip("hybrid_rewind", f"could not load {model_id}")
    if eng._trimmable or not eng._pld_hybrid:
        return skip("hybrid_rewind", "not a hybrid cache; rewind path won't engage")
    eng.prompt_lookup = False   # plain decode path; rewind is what's under test

    messages = [
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "Print hello world in Python."},
    ]
    prompt_ids = list(eng.tok.apply_chat_template(messages, add_generation_prompt=True))

    # Turn 1: real generation — cache ends at prompt + gen ids, snapshot at prompt end.
    eng._reset_cache()
    _t1, _s1 = eng.generate(list(prompt_ids), max_tokens=24)
    check("turn 1 left a rewind snapshot", eng._rewind_snap is not None)
    snap_pos = eng._rewind_snap["pos"]
    gen_ids = eng._cached_ids[len(prompt_ids):]
    check("turn 1 cache holds prompt+generation",
          len(gen_ids) > 4, f"gen_ids={len(gen_ids)}")

    # Turn 2 prompt: keep a PREFIX of the generation, then diverge (the truncated-turn
    # re-render shape: same start, different tail).
    keep = len(gen_ids) // 2
    tail = list(eng.tok.encode("\nNow say done.", add_special_tokens=False))
    recover = list(prompt_ids) + list(gen_ids[:keep]) + tail

    rewinds = []
    orig_rewind = eng._rewind_to
    eng._rewind_to = lambda t, u: (rewinds.append(u), orig_rewind(t, u))[1]
    warm_text, warm_stats = eng.generate(list(recover), max_tokens=24)
    check("rewind path actually ran", rewinds == [len(prompt_ids) + keep], rewinds)
    check("warm run reused the shared prefix through the divergence",
          warm_stats.cached_tokens == len(prompt_ids) + keep,
          f"cached={warm_stats.cached_tokens} want={len(prompt_ids) + keep} "
          f"(snap@{snap_pos})")

    # Baseline: same prompt, fresh engine state.
    eng._rewind_to = orig_rewind
    eng._reset_cache()
    fresh_text, _ = eng.generate(list(recover), max_tokens=24)
    if _is_quantized(model_id):
        # quantized: argmax exact-ties → outputs may differ legally; both must decode.
        check("quantized: warm rewind generation non-empty", len(warm_text) > 0)
        check("quantized: fresh generation non-empty", len(fresh_text) > 0)
        print(f"  (quantized weights: strict equality skipped; warm={warm_text[:40]!r} "
              f"fresh={fresh_text[:40]!r})")
    else:
        # CORRUPTION GUARD: rewound-cache generation == fresh generation, byte-exact.
        # If this fails the rewind corrupted the cache — do NOT loosen; STOP and report.
        check("hybrid rewind: warm generation == fresh generation",
              warm_text == fresh_text,
              f"\n--- WARM (rewound) ---\n{warm_text!r}\n--- FRESH ---\n{fresh_text!r}")


# === Tier 2b: cache-reuse correctness on the truncation/degenerate paths ==
# These force the two rare cache-fast-path branches and assert
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


# === Tier 2c: cache quarantine push/pop bit-exactness ==============
# The subagent/Task tool runs a sub-agent on a QUARANTINED cache: engine.push_cache
# stashes the main session's warm cache aside, the sub-agent runs on a fresh one, and
# pop_cache restores the main cache. The invariant that makes this safe on the
# non-trimmable hybrid is that after pop the main cache generates BYTE-IDENTICALLY to a
# never-pushed control — i.e. push/pop is a lossless snapshot/restore. This exercises it
# on the small trimmable model (push/pop is model-agnostic); a bug that corrupts the
# restored cache breaks the strict equality below — do NOT loosen.

def test_push_pop_bit_exact():
    if os.environ.get("CHAD_FAST_TESTS"):
        return skip("push_pop_bit_exact", "CHAD_FAST_TESTS set (fast gate)")
    # push/pop is model-agnostic (RAM-tuple stash + optional disk spill); the trimmable
    # default is the CI path. A run on the real hybrid (bf16 Ornith) is also wanted,
    # so CHAD_TEST_HYBRID_MODEL — when set — points THIS test at those weights.
    model_id = os.environ.get("CHAD_TEST_HYBRID_MODEL", TIER2_MODEL)
    eng = _build_engine(model_id=model_id)
    if eng is None:
        return skip("push_pop_bit_exact", f"could not load {model_id}")

    def _tmpl(user):
        return list(eng.tok.apply_chat_template(
            [{"role": "system", "content": "You are a precise coding assistant."},
             {"role": "user", "content": user}], add_generation_prompt=True))

    P = _tmpl("Write a one-line Python function that squares n.")
    Q = _tmpl("Explain in one sentence what a hash map is.")  # unrelated sub-agent churn

    # Control: reach state S (prime the cache with P + a short generation), snapshot the
    # resident ids, then continue from S with an extended prompt — NO push in between.
    eng._reset_cache()
    eng.generate(list(P), max_tokens=8)
    ids_s = list(eng._cached_ids)
    P2 = ids_s + list(P[:4])                       # extend the cache with a few valid tokens
    control, _ = eng.generate(list(P2), max_tokens=24)

    # Quarantine: reach the SAME state S, push it aside, run an unrelated generation on a
    # fresh cache (the sub-agent), pop, then continue from the restored S with the same P2.
    eng._reset_cache()
    eng.generate(list(P), max_tokens=8)
    check("re-reached state S deterministically", eng._cached_ids == ids_s,
          f"{len(eng._cached_ids)} vs {len(ids_s)}")
    eng.push_cache()
    check("push handed out a fresh empty cache", eng._cached_ids == [], eng._cached_ids)
    eng.generate(list(Q), max_tokens=8)            # sub-agent work in the quarantined cache
    eng.pop_cache()
    check("pop restored _cached_ids to state S exactly", eng._cached_ids == ids_s,
          f"{len(eng._cached_ids)} vs {len(ids_s)}")
    quar, _ = eng.generate(list(P2), max_tokens=24)

    # CORRUPTION GUARD: the post-pop continuation must equal the never-pushed control,
    # byte-for-byte. If this fails, push/pop corrupted the cache — STOP and report.
    check("post-pop generation == never-pushed control", quar == control,
          f"\n--- QUARANTINE ---\n{quar!r}\n--- CONTROL ---\n{control!r}")


# --- On-disk KV cache bounding (no model, pure filesystem) ---------

def _touch(path, size, age_s=0.0):
    """Create a file of `size` bytes whose mtime is `age_s` seconds in the past."""
    import time
    with open(path, "wb") as f:
        f.write(b"\0" * size)
    t = time.time() - age_s
    os.utime(path, (t, t))
    return path


def test_sweep_orphan_spills_removes_only_old_push_files():
    import tempfile

    from chad.engine import sweep_orphan_spills
    with tempfile.TemporaryDirectory() as d:
        old_push = _touch(os.path.join(d, "push-aaa.safetensors"), 100, age_s=10 * 3600)
        new_push = _touch(os.path.join(d, "push-bbb.safetensors"), 100, age_s=0)
        old_warm = _touch(os.path.join(d, "warm-ccc.safetensors"), 100, age_s=10 * 3600)
        freed = sweep_orphan_spills(d, max_age_s=6 * 3600)
        check("old push-spill removed", not os.path.exists(old_push))
        check("fresh push-spill kept", os.path.exists(new_push))
        check("old warm file untouched by sweep", os.path.exists(old_warm))
        check("sweep reports bytes freed", freed == 100, f"freed={freed}")
    check("sweep of a missing dir is a no-op",
          sweep_orphan_spills("/nonexistent/nope", max_age_s=1) == 0)


def test_enforce_cache_budget_evicts_lru_first():
    import tempfile

    from chad.engine import enforce_cache_budget
    with tempfile.TemporaryDirectory() as d:
        # 4 x 100B, oldest first; budget 250B -> the two oldest must go.
        paths = [_touch(os.path.join(d, f"warm-{i}.safetensors"), 100,
                        age_s=(4 - i) * 3600) for i in range(4)]
        freed = enforce_cache_budget(d, max_bytes=250, protect=set())
        check("evicted down to budget", freed == 200, f"freed={freed}")
        check("oldest two evicted",
              [os.path.exists(p) for p in paths] == [False, False, True, True],
              str([os.path.exists(p) for p in paths]))


def test_enforce_cache_budget_protects_paths():
    import tempfile

    from chad.engine import enforce_cache_budget
    with tempfile.TemporaryDirectory() as d:
        oldest = _touch(os.path.join(d, "warm-old.safetensors"), 100, age_s=2 * 3600)
        newest = _touch(os.path.join(d, "warm-new.safetensors"), 100, age_s=0)
        # Budget forces one eviction; the LRU candidate is protected, so the next
        # oldest goes instead — protection wins even if the dir stays over budget.
        enforce_cache_budget(d, max_bytes=100, protect={oldest})
        check("protected LRU file survives", os.path.exists(oldest))
        check("unprotected newer file evicted instead", not os.path.exists(newest))


def test_enforce_cache_budget_disabled_when_zero():
    import tempfile

    from chad.engine import enforce_cache_budget
    with tempfile.TemporaryDirectory() as d:
        p = _touch(os.path.join(d, "warm-x.safetensors"), 100, age_s=3600)
        check("zero budget is disabled", enforce_cache_budget(d, 0, set()) == 0)
        check("zero budget removes nothing", os.path.exists(p))


def test_ckpt_path_filenames_are_kind_tagged():
    import tempfile

    from chad.engine import Engine
    with tempfile.TemporaryDirectory() as d:
        eng = object.__new__(Engine)  # bypass __init__ (no weights to load)
        eng.model_id = "test-model"
        eng.cache_dir = d
        warm = os.path.basename(eng._ckpt_path([1, 2, 3]))
        push = os.path.basename(eng._ckpt_path([1, 2, 3], tag="push"))
        check("warm checkpoint basename tagged", warm.startswith("warm-"), warm)
        check("push checkpoint basename tagged", push.startswith("push-"), push)
        check("kinds hash differently", warm.split("-", 1)[1] != push.split("-", 1)[1])


def test_adaptive_chunk_bounds():
    """Tier 1 (no weights): the adaptive prefill chunk obeys its
    contract at the boundaries that don't depend on live memory numbers:

      - dense model, short context -> the 512 base (the measured-flat 9B keeps its
        old default; also what the interrupted-prefill test's chunk math assumes);
      - MoE model, short context -> the 2048 base (+14% measured prefill);
      - ANY model at huge resident context -> the 256 floor (the unfused
        head_dim-256 score tensor makes big chunks the long-context OOM vector; the
        cap math cannot exceed the floor at kv=3M on any real Mac's free band).
    """
    from chad.engine import Engine

    eng = object.__new__(Engine)  # bypass __init__ (no weights)
    # No _is_moe/_n_attn_heads set: getattr defaults (dense, 16 heads) apply — the
    # same shape a fake-engine test double presents.
    # kv=50M: the score-tensor cap math would need a >1.6 TB free band to beat the
    # floor, so this asserts the floor on any real machine, laptop or Studio.
    check("dense short-ctx chunk is the 512 base", eng._adaptive_chunk(0) == 512,
          eng._adaptive_chunk(0))
    check("huge-ctx chunk hits the 256 floor", eng._adaptive_chunk(50_000_000) == 256,
          eng._adaptive_chunk(50_000_000))
    eng._is_moe = True
    eng._n_attn_heads = 16
    got = eng._adaptive_chunk(0)
    check("MoE short-ctx chunk is the 2048 base", got == 2048, got)
    check("MoE huge-ctx chunk hits the 256 floor too",
          eng._adaptive_chunk(50_000_000) == 256)


def test_load_fails_fast_without_mlx():
    """Tier 1 (no weights): when the mlx imports failed (broken/half-installed
    mlx-metal, or a non-Apple host that somehow built an Engine), load() must raise
    a RuntimeError naming the original import cause — NOT fall through to `load(path)`
    and die with a bare `TypeError: 'NoneType' object is not callable` 300 lines from
    the real problem. Regression for the missing-libmlx.dylib dogfood incident."""
    from chad import engine as engmod
    from chad.engine import Engine

    eng = object.__new__(Engine)  # bypass __init__ (no weights)
    orig_has, orig_err = engmod._HAS_MLX, engmod._MLX_IMPORT_ERROR
    cause = ImportError("dlopen: libmlx.dylib not loaded")
    try:
        engmod._HAS_MLX = False
        engmod._MLX_IMPORT_ERROR = cause
        try:
            eng.load()
            check("load() without mlx raises", False, "no exception raised")
        except RuntimeError as e:
            check("load() without mlx raises RuntimeError, not NoneType TypeError",
                  "libmlx.dylib" in str(e) and "mlx-metal" in str(e), str(e))
            check("original import error is chained as __cause__",
                  e.__cause__ is cause, repr(e.__cause__))
        except TypeError as e:
            check("load() without mlx must NOT raise the opaque NoneType TypeError",
                  False, str(e))
    finally:
        engmod._HAS_MLX, engmod._MLX_IMPORT_ERROR = orig_has, orig_err


def test_prefill_oom_retry_rolls_back():
    """Tier 1 (no weights): a Metal OOM inside a prefill chunk (catchable on
    mlx>=0.32) must (a) restore every cache layer's state to the pre-chunk
    snapshot, (b) retry at half the width, and (c) still feed every token exactly
    once. A non-memory RuntimeError must propagate unchanged."""
    import mlx.core as mx

    from chad.engine import Engine

    class _FakeCacheItem:
        def __init__(self):
            self.state = mx.array([0.0])

    forwards = []          # width of every attempted forward
    restored = []          # state objects assigned back by _restore_cache_refs

    class _TrackedItem(_FakeCacheItem):
        def __setattr__(self, name, value):
            if name == "state" and "state" in self.__dict__:
                restored.append(value)
            super().__setattr__(name, value)

    def _model(arr, cache=None):
        w = int(arr.shape[1])
        forwards.append(w)
        if w > 64:
            raise RuntimeError("[METAL] Command buffer execution failed: "
                               "Insufficient Memory")

    eng = object.__new__(Engine)
    eng._cache = [_TrackedItem(), _TrackedItem()]
    eng.model = _model
    eng._cached_ids = []

    fed = eng._prefill(list(range(200)))  # chunk=None -> adaptive base 512
    ok_widths = [w for w in forwards if w <= 64]
    check("OOM retry fed the full range", fed == 200, fed)
    check("first attempt was the adaptive base (dense=512 -> capped to 200)",
          forwards[0] == 200, forwards)
    check("failures halved the width down to <=64",
          all(w <= 64 for w in forwards[2:]), forwards)
    check("every token fed exactly once by the successful widths",
          sum(ok_widths) == 200, forwards)
    check("cache state was rolled back once per failed chunk",
          len(restored) == 2 * 2, f"{len(restored)} restores (2 layers x 2 fails)")

    # Non-memory errors propagate: the retry logic must not swallow real bugs.
    def _boom(arr, cache=None):
        raise RuntimeError("boom")
    eng2 = object.__new__(Engine)
    eng2._cache = [_FakeCacheItem()]
    eng2.model = _boom
    eng2._cached_ids = []
    try:
        eng2._prefill([1, 2, 3])
        check("non-memory RuntimeError propagates", False, "no exception raised")
    except RuntimeError as e:
        check("non-memory RuntimeError propagates", "boom" in str(e), str(e))


def test_bounded_rewind_orchestration():
    """Tier 1 (no weights): `_sync_to`'s bounded-rewind branch — the
    orchestration math, with the tier-2-proven primitives (restore recurrent /
    trim KV / re-feed) stubbed out and recorded.

      cache: [P0..P9, G0..G4]  (prompt + last turn's generation), snapshot@10
      target: [P0..P9, G0, G1, X, Y]  (divergence INSIDE the last turn, common=12)

    Expected: restore recurrent once, trim the attention KV back to the snapshot
    (peel = 15-10 = 5), re-feed exactly the agreed-on tokens target[10:12], and
    report common=12 — never a cache reset. Then: a divergence BEFORE the snapshot
    must fall back to the full rebuild (reset + warm-prefix reload), and a missing
    snapshot must too."""
    from chad.engine import Engine

    class _NoTrim:
        def is_trimmable(self):
            return False

    def make(snap_pos):
        eng = object.__new__(Engine)
        eng._pld_hybrid = True
        eng._trimmable = False
        eng.kv_bits = None
        eng.draft = None
        eng._cache = [_NoTrim()]
        eng.model = type("M", (), {"layers": [0]})()  # _sub_caches slices by layer count
        eng._cached_ids = list(range(100, 110)) + list(range(200, 205))  # P0..P9+G0..G4
        eng._rewind_snap = ({"pos": snap_pos, "recurrent": "SNAP"}
                            if snap_pos is not None else None)
        eng.calls = []
        eng._restore_recurrent = lambda s: eng.calls.append(("restore", s))
        eng._trim_kv = lambda n: eng.calls.append(("trim", n))
        eng._prefill = lambda ids, *a, **k: (eng.calls.append(("feed", list(ids))),
                                             len(ids))[1]
        eng._reset_cache = lambda: eng.calls.append(("reset",))
        eng._reload_warm_prefix = lambda t: (eng.calls.append(("warm",)), 0)[1]
        return eng

    # Divergence inside the last turn: rewind path, no reset.
    eng = make(snap_pos=10)
    target = list(range(100, 110)) + [200, 201, 999, 998]   # common with cache = 12
    common = eng._sync_to(target)
    check("rewind: common is the shared prefix", common == 12, common)
    check("rewind: restore+trim+feed, in order, no reset",
          eng.calls == [("restore", "SNAP"), ("trim", 5), ("feed", [200, 201])],
          eng.calls)
    check("rewind: cached_ids land at target[:common]",
          eng._cached_ids == target[:12], eng._cached_ids)
    check("rewind: snapshot survives for reuse", eng._rewind_snap is not None)

    # Divergence BEFORE the snapshot (e.g. compaction): full rebuild fallback.
    eng = make(snap_pos=10)
    early = [100, 101, 777]                                  # common = 2 < snap pos
    common = eng._sync_to(early)
    check("pre-snapshot divergence falls back to rebuild",
          ("reset",) in eng.calls and ("warm",) in eng.calls, eng.calls)
    check("pre-snapshot divergence: no rewind primitives ran",
          not any(c[0] in ("restore", "trim") for c in eng.calls), eng.calls)

    # No snapshot at all: rebuild, exactly as before the bounded-rewind work.
    eng = make(snap_pos=None)
    eng._sync_to(target)
    check("no snapshot -> rebuild", ("reset",) in eng.calls, eng.calls)

    # Snapshot exactly at the divergence point: trim only, nothing re-fed.
    eng = make(snap_pos=12)
    common = eng._sync_to(target)
    check("snapshot at divergence: trim only, no feed",
          eng.calls == [("restore", "SNAP"), ("trim", 3)], eng.calls)
    check("snapshot at divergence: common intact", common == 12, common)


def test_take_rewind_snapshot_gating():
    """Tier 1: the snapshot is only taken on the validated cache composition —
    hybrid (_pld_hybrid) with tokens resident. Anything else is a silent no-op
    (a snapshot on the wrong cache kind would CORRUPT a later rewind: _trim_kv
    only trims KVCache/QuantizedKVCache layers). The quantized
    cache is built quantized-from-start (no mid-decode type conversion), so
    kv_bits no longer disables the snapshot — the quantized path is covered by
    test_engine_kvquant.py's end-to-end rewind."""
    from chad.engine import Engine

    def make(pld_hybrid=True, kv_bits=None):
        eng = object.__new__(Engine)
        eng._pld_hybrid = pld_hybrid
        eng.kv_bits = kv_bits
        eng._rewind_snap = None
        eng._snap_recurrent = lambda: "REC"
        return eng

    eng = make()
    eng._take_rewind_snapshot(42)
    check("hybrid snapshot taken with explicit pos",
          eng._rewind_snap == {"pos": 42, "recurrent": "REC"}, eng._rewind_snap)
    eng = make(pld_hybrid=False)
    eng._take_rewind_snapshot(42)
    check("non-hybrid: no snapshot", eng._rewind_snap is None)
    eng = make(kv_bits=8)
    eng._take_rewind_snapshot(42)
    check("kv_bits (quantized-native): snapshot taken",
          eng._rewind_snap == {"pos": 42, "recurrent": "REC"}, eng._rewind_snap)
    eng = make()
    eng._take_rewind_snapshot(0)
    check("zero resident tokens: no snapshot", eng._rewind_snap is None)


def test_snapshot_survives_empty_kvcache():
    """Regression (caught on real weights, missed by simple fakes): mlx-lm's
    KVCache.state property RAISES on an EMPTY cache (`keys` is None before the
    first token), so the pre-chunk snapshot must read raw attributes, never the
    property. A snapshot/restore round-trip on an empty-KVCache-like item must not
    raise, must restore attribute values exactly, and must deep-copy list attrs one
    level (ArraysCache mutates its list elements in place)."""
    from chad.engine import Engine

    class _EmptyKVLike:
        def __init__(self):
            self.keys = None
            self.values = None
            self.offset = 0

        @property
        def state(self):  # what mlx-lm's KVCache does when empty
            return self.keys.shape  # AttributeError on None

    class _ArraysLike:
        def __init__(self):
            self.cache = ["a", "b"]

    eng = object.__new__(Engine)
    kv, arr = _EmptyKVLike(), _ArraysLike()
    eng._cache = [kv, arr]
    snap = eng._snapshot_cache_refs()          # must not touch .state
    kv.keys, kv.offset = "poisoned", 99        # simulate a failed chunk's leftovers
    arr.cache[0] = "poisoned"                  # in-place list element mutation
    eng._restore_cache_refs(snap)
    check("empty-KVCache attrs restored", kv.keys is None and kv.offset == 0,
          (kv.keys, kv.offset))
    check("list attr restored one level deep", arr.cache == ["a", "b"], arr.cache)


def test_keyed_sampler_worker_thread_entropy():
    """Regression (canary): MLX's implicit-key RNG never advances on
    non-main threads, so make_sampler(temp>0) run from an engine worker
    thread replays ONE frozen noise vector — byte-identical responses for
    identical prompts and loop-prone quasi-greedy decoding. _KeyedSampler must
    (a) vary across draws on a worker thread, (b) replay exactly for a fixed
    seed, (c) diverge across seeds."""
    import threading

    try:
        import mlx.core as mx
    except ImportError:
        skip("keyed sampler", "mlx not installed")
        return
    from chad.engine import _KeyedSampler

    logits = mx.array([[1.0, 1.1, 0.9, 1.05, 0.95, 1.2, 0.8, 1.15]])
    out = {}

    def draws(name, seed):
        s = _KeyedSampler(1.0, seed=seed)
        out[name] = [int(s(logits).item()) for _ in range(12)]

    for name, seed in (("a", 1), ("b", 2), ("a2", 1)):
        t = threading.Thread(target=draws, args=(name, seed))
        t.start()
        t.join()
    check("worker-thread draws vary (frozen-noise regression)",
          len(set(out["a"])) > 1, out["a"])
    check("fixed seed replays exactly", out["a"] == out["a2"], (out["a"], out["a2"]))
    check("seeds diverge", out["a"] != out["b"], (out["a"], out["b"]))


def test_keyed_sampler_min_p_trims_sub_floor_tail():
    """Min_p must actually bind inside `_KeyedSampler` (the temp>0 hot
    path TB2 uses — `make_sampler`'s own min_p is dead code there, see engine.py's
    guarded import comment). Token 1 (logit -3.0) sits below the min_p=0.1 floor
    relative to token 0's max logit (threshold = 0 + ln(0.1) ~= -2.3) but is NOT
    astronomically rare (~4.5% unfiltered) — over 300 draws it would show up with
    near-certainty if the filter weren't binding, so a clean absence is a real
    signal, not a coincidence of small-sample luck."""
    try:
        import mlx.core as mx
    except ImportError:
        skip("keyed sampler min_p", "mlx not installed")
        return
    from chad.engine import _KeyedSampler

    logits = mx.array([[0.0, -3.0]])
    s = _KeyedSampler(1.0, seed=7, min_p=0.1)
    draws = [int(s(logits).item()) for _ in range(300)]
    check("sub-floor token never drawn under min_p", 1 not in draws, set(draws))
    check("top token still reachable", 0 in draws, set(draws))


def test_keyed_sampler_top_p_trims_tail():
    """Analogous to the min_p test above, for top_p: with probs [0.9, 0.05, 0.05]
    and top_p=0.9, only the dominant token survives nucleus filtering (verified
    directly against mlx_lm.sample_utils.apply_top_p's semantics)."""
    try:
        import mlx.core as mx
        import numpy as np
    except ImportError:
        skip("keyed sampler top_p", "mlx not installed")
        return
    from chad.engine import _KeyedSampler

    probs = np.array([0.9, 0.05, 0.05])
    logits = mx.array([np.log(probs).tolist()])
    s = _KeyedSampler(1.0, seed=11, top_p=0.9)
    draws = [int(s(logits).item()) for _ in range(200)]
    check("only the dominant token survives top_p nucleus", set(draws) == {0}, set(draws))


def test_keyed_sampler_default_off_is_full_support():
    """Regression guard (086/084-style): _KeyedSampler with no min_p/top_p (the
    shipped default, 0.0 == no-op) must sample the FULL support — including a
    low-probability-but-nonzero token that a bound min_p=0.1 would exclude (see
    test_keyed_sampler_min_p_trims_sub_floor_tail's same logit vector). If this
    regresses to filtering by default, TB2's on-policy distribution silently
    narrows with no operator opt-in."""
    try:
        import mlx.core as mx
    except ImportError:
        skip("keyed sampler default-off", "mlx not installed")
        return
    from chad.engine import _KeyedSampler

    logits = mx.array([[0.0, -3.0]])
    s = _KeyedSampler(1.0, seed=7)  # min_p=0.0, top_p=0.0 defaults
    draws = [int(s(logits).item()) for _ in range(300)]
    check("low-prob tail token still reachable with filters off", 1 in draws, set(draws))


def test_make_sampler_argmax_unchanged_by_min_p_top_p():
    """`Engine.generate`'s kwargs construction picks
    `make_sampler(temp=0.0, ...)` on the greedy path regardless of min_p/top_p
    (engine.py ~1073). `make_sampler` itself short-circuits to a pure argmax
    lambda when temp==0 (see mlx_lm.sample_utils.make_sampler), so passing
    nonzero min_p/top_p through on that call must not change the argmax output.
    Tested directly against `make_sampler` (the actual temp==0 code path) rather
    than mocking `Engine.generate`."""
    try:
        import mlx.core as mx
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        skip("make_sampler argmax parity", "mlx not installed")
        return

    logits = mx.array([[1.0, 5.0, 2.0]])
    plain = make_sampler(temp=0.0)
    filtered = make_sampler(temp=0.0, min_p=0.5, top_p=0.1)
    check("argmax unaffected by min_p/top_p at temp=0",
          int(plain(logits).item()) == int(filtered(logits).item()),
          (int(plain(logits).item()), int(filtered(logits).item())))
    check("argmax picks the actual max logit", int(plain(logits).item()) == 1,
          int(plain(logits).item()))


if __name__ == "__main__":
    tier1 = (test_longest_match_wins, test_recency_tie_break, test_no_match,
             test_guards, test_draft_length, test_prefill_progress_callback,
             test_interrupted_prefill_records_only_fed_tokens,
             test_stop_condition_soft_close,
             test_sweep_orphan_spills_removes_only_old_push_files,
             test_enforce_cache_budget_evicts_lru_first,
             test_enforce_cache_budget_protects_paths,
             test_enforce_cache_budget_disabled_when_zero,
             test_ckpt_path_filenames_are_kind_tagged,
             test_adaptive_chunk_bounds,
             test_load_fails_fast_without_mlx,
             test_prefill_oom_retry_rolls_back,
             test_snapshot_survives_empty_kvcache,
             test_bounded_rewind_orchestration,
             test_take_rewind_snapshot_gating,
             test_keyed_sampler_worker_thread_entropy,
             test_keyed_sampler_min_p_trims_sub_floor_tail,
             test_keyed_sampler_top_p_trims_tail,
             test_keyed_sampler_default_off_is_full_support,
             test_make_sampler_argmax_unchanged_by_min_p_top_p)
    tier2 = (test_pld_equals_greedy,
             test_pld_hybrid_equals_greedy,
             test_hybrid_rewind_matches_fresh,
             test_degenerate_reprefill_matches_fresh,
             test_truncation_recovery_matches_fresh,
             test_push_pop_bit_exact)
    for fn in tier1 + tier2:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            FAIL += 1
            print(f"  ERROR in {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{PASS} passed, {FAIL} failed, {SKIP} skipped")
    raise SystemExit(1 if FAIL else 0)
