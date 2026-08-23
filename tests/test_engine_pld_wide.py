"""Bit-exactness of the wide prompt-lookup path (_generate_pld_wide).

Same synthetic-hybrid strategy as test_engine_pld_hybrid.py (see that module
docstring for why the tiny model stays unquantized and why drafts are
injected): the invariant is "greedy wide-PLD reproduces plain greedy decoding
token for token, whatever the drafter proposes". The wide path adds two
states the old path does not have — a pipelined LAZY arm (cold decode with
the sampled token one step behind) and a COMMITTED arm (verify-to-verify
chaining inside a span) — plus the d[0] evidence gate on the lazy->span
transition. Each mode below pins one of them:

  none     -- lookup never matches: pure pipelined cold decode
  correct  -- every draft right: verify chains in the COMMITTED state
  partial  -- 2 right + bogus tail: capture-replay/re-feed rollback every step
  wrong    -- all bogus: total rejection every step
  late     -- cold for 5 tokens, then correct: the LAZY->span entry, where
              the draft only fires if d[0] equals the materialized sample
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from chad import engine as eng_mod  # noqa: E402
from test_engine_pld_hybrid import (  # noqa: E402
    BOGUS,
    N_TOKENS,
    PROMPT,
    _engine,
    _greedy,
    tiny,  # noqa: F401  (fixture)
)

WIDE = 8


def _wide_engine(model):
    eng = _engine(model)
    eng.pld_wide = True
    eng.pld_wide_draft = WIDE
    eng.pld_wide_ngram = 3
    eng.pld_wide_min_ngram = 1
    eng.pld_wide_min_draft = 2
    return eng


def _run(mode, model, reference, stop_condition=None):
    prompt_len = len(PROMPT)

    def fake_lookup(arr, n, num_draft, ngram_max, ngram_min):
        gen = n - prompt_len
        if mode == "none" or (mode == "late" and gen < 5):
            return [], 0
        actual = reference[gen : gen + num_draft]
        if len(actual) < num_draft:
            return [], 0
        if mode in ("correct", "late"):
            return list(actual), ngram_max
        if mode == "partial":
            return list(actual[:2]) + [BOGUS] * (num_draft - 2), ngram_max
        return [BOGUS] * num_draft, ngram_max

    real = eng_mod.prompt_lookup_draft_arr
    eng_mod.prompt_lookup_draft_arr = fake_lookup
    try:
        eng = _wide_engine(model)
        text, stats = eng._generate_pld_wide(PROMPT, N_TOKENS, None, None,
                                             stop_condition=stop_condition)
    finally:
        eng_mod.prompt_lookup_draft_arr = real
    return [int(t) for t in text.split()], stats


def _diverge(ref, got):
    return next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b),
                min(len(ref), len(got)))


def test_cold_pipeline_is_bit_exact(tiny):
    """No matches anywhere: the pipelined lazy arm alone must equal plain
    greedy — this is the arm every novel-text token rides."""
    ref = _greedy(tiny, PROMPT, N_TOKENS)
    got, stats = _run("none", tiny, ref)
    assert stats.draft_proposed == 0
    assert stats.forwards == 0, "no verify forward should have run"
    assert got == ref, f"diverged at {_diverge(ref, got)}"


def test_fully_accepted_wide_drafts_are_bit_exact(tiny):
    ref = _greedy(tiny, PROMPT, N_TOKENS)
    got, stats = _run("correct", tiny, ref)
    assert stats.draft_accepted > 0
    assert stats.draft_accepted == stats.draft_proposed
    assert got == ref, f"diverged at {_diverge(ref, got)}"


def test_wide_verify_covers_many_tokens_per_forward(tiny):
    """The economics: chained correct verifies must land ~WIDE+1 tokens per
    verify forward, or the COMMITTED-state chaining is broken and every span
    pays an interleaved plain step."""
    ref = _greedy(tiny, PROMPT, N_TOKENS)
    got, stats = _run("correct", tiny, ref)
    assert got == ref
    per = len(got) / max(1, stats.forwards)
    assert per >= WIDE * 0.8, (per, stats.forwards)


def test_partial_rejection_is_bit_exact(tiny):
    ref = _greedy(tiny, PROMPT, N_TOKENS)
    got, stats = _run("partial", tiny, ref)
    assert stats.draft_accepted > 0
    assert stats.draft_proposed > stats.draft_accepted, "nothing rejected"
    assert got == ref, f"diverged at {_diverge(ref, got)}"


def test_total_rejection_is_bit_exact(tiny):
    ref = _greedy(tiny, PROMPT, N_TOKENS)
    got, stats = _run("wrong", tiny, ref)
    assert stats.draft_proposed > 0
    assert stats.draft_accepted == 0
    assert got == ref, f"diverged at {_diverge(ref, got)}"


def test_stop_condition_fires_on_cold_path(tiny):
    """Regression: the pipelined cold arm once `continue`d past the shared
    stop_condition check, so a cold generation NEVER stopped at the agent's
    tool-call boundary and every step ran to max_tokens (measured as a
    40-minute eval task). The cold arm must honor stop_condition with the
    plain path's per-token latency."""
    ref = _greedy(tiny, PROMPT, N_TOKENS)
    got, stats = _run("none", tiny, ref,
                      stop_condition=lambda text, n: n >= 7)
    assert stats.stop_condition_fired
    assert stats.generated_tokens == 7, stats.generated_tokens
    assert got == ref[:7]


def test_stop_condition_fires_after_wide_verify(tiny):
    """Inside a span the check runs per committed batch (documented lag of
    at most one draft width), but it must still fire and stop the turn."""
    ref = _greedy(tiny, PROMPT, N_TOKENS)
    got, stats = _run("correct", tiny, ref,
                      stop_condition=lambda text, n: n >= 5)
    assert stats.stop_condition_fired
    assert stats.generated_tokens < N_TOKENS
    assert got == ref[: len(got)]


def test_lazy_to_span_entry_is_bit_exact(tiny):
    """Cold pipeline for 5 tokens, then a span appears: the entry must sync
    the in-flight token, confirm it against d[0], and verify d[1:] — the
    one place the pipelined and speculative arms hand off."""
    ref = _greedy(tiny, PROMPT, N_TOKENS)
    got, stats = _run("late", tiny, ref)
    assert stats.draft_accepted > 0, "span entry never fired"
    assert got == ref, f"diverged at {_diverge(ref, got)}"
