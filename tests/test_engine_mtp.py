"""MTP self-speculative decoding: distribution safety and cache bookkeeping.

The MTP path's core promise is the same as PLD's: whatever the drafter
proposes, the committed output must be EXACTLY what plain decoding would have
produced (greedy: bit-equal tokens; sampled: the same distribution via
rejection sampling). A random-weight head drafts garbage, so it exercises the
full-rollback arm (restore recurrent + trim KV + re-feed + drafter-cache trim)
on every step; a zeroed model makes drafter and verifier agree trivially, so
it exercises the full-accept arm and the drafter catch-up bookkeeping.

These run on the tiny synthetic qwen3_5 hybrid from the PLD suite —
unquantized, so exact argmax comparisons are meaningful (see that module's
docstring for why quantized grids make greedy ties ambiguous)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models import cache as cache_utils  # noqa: E402

from chad import mlx_mtp  # noqa: E402
from chad.engine import Engine  # noqa: E402
from test_mlx_fastpath import TINY_CFG  # noqa: E402

PROMPT = [5, 9, 11, 22, 33, 44, 55, 66, 77, 88, 101, 102, 7]
N_TOKENS = 40


def _build_tiny(seed=0, zero=False):
    import copy

    from mlx.utils import tree_map
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    cfg = copy.deepcopy(TINY_CFG)
    cfg["text_config"]["head_dim"] = 64
    mx.random.seed(seed)
    model = Model(ModelArgs.from_dict(cfg))
    if zero:
        model.update(tree_map(mx.zeros_like, model.parameters()))
    model.eval()
    return model


class _Detok:
    def __init__(self):
        self.reset()

    def reset(self):
        self.text = ""
        self.last_segment = ""

    def add_token(self, t):
        self.last_segment = f" {t}"
        self.text += self.last_segment

    def finalize(self):
        pass


class _Tok:
    eos_token_id = None
    eos_token_ids = []

    def __init__(self):
        self.detokenizer = _Detok()


def _engine(model, head, temp=0.0, num_draft=4, adaptive=False):
    # adaptive=False by default: these tests pin the fixed-width contract
    # (forwards-count assertions assume a constant k), and the synthetic
    # stubs emit flat logits whose zero top-2 margin legitimately clamps
    # the adaptive schedule shallow. Adaptive coverage lives in the
    # test_adaptive_* tests below.
    eng = object.__new__(Engine)
    eng.model = model
    eng.tok = _Tok()
    eng.kv_bits = None
    eng.temp = temp
    eng.mtp_adaptive = adaptive
    eng.min_p = eng.top_p = 0.0
    eng.prompt_lookup = False
    eng.pld_num_draft = 8
    eng.pld_ngram = 3
    eng.enable_pld_hybrid = False
    eng.mtp_num_draft = num_draft
    eng._mtp_head = head
    eng.cache_dir = None
    eng.max_context = None
    eng.effective_ctx = 4096
    eng.model_id = "tiny-hybrid"
    eng._model_path = ""
    eng._cached_ids = []
    eng._rewind_snap = None
    eng._cache_stack = []
    eng._warm_prefix_ids = None
    eng.kv_bytes_per_token = 0.0
    eng._is_moe = False
    eng._reset_cache()
    return eng


def _greedy(model, prompt_ids, n):
    cache = cache_utils.make_prompt_cache(model)
    logits = model(mx.array(prompt_ids)[None], cache=cache)
    out = [int(mx.argmax(logits[0, -1]).item())]
    for _ in range(n - 1):
        logits = model(mx.array([out[-1]])[None], cache=cache)
        out.append(int(mx.argmax(logits[0, -1]).item()))
    return out


def _head_for(model):
    mx.random.seed(1234)  # random weights: drafts will be near-always wrong
    head = mlx_mtp.build(model.language_model.args)
    head.eval()
    return head


def test_random_head_greedy_is_bit_exact():
    """Garbage drafts must cost nothing but time: full rollback every step."""
    model = _build_tiny()
    ref = _greedy(model, PROMPT, N_TOKENS)
    eng = _engine(model, _head_for(model))
    text, stats = eng._generate_mtp(PROMPT, N_TOKENS, None, None)
    got = [int(t) for t in text.split()]
    assert stats.draft_proposed > 0, "drafting never ran"
    assert got == ref, (
        f"diverged at {next(i for i, (a, b) in enumerate(zip(ref, got)) if a != b)}")
    # fed_ids invariant: everything but the pending token is resident.
    assert len(eng._cached_ids) == len(PROMPT) + len(got) - 1


def test_random_head_greedy_bit_exact_through_capture_replay():
    """Same bit-exactness contract, but through the fastpath's
    capture-and-replay arm: with install() applied the verify forward records
    the GDN recurrence inputs and every rejection rebuilds the linear caches
    by replaying them over the accepted prefix. A random head rejects nearly
    every draft, so this hammers the replay; output must still be bit-equal
    to plain decoding."""
    import mlx.nn as nn

    from chad import mlx_fastpath
    model = _build_tiny()
    # install() engages only on quantized hybrids; quantize THEN take the
    # reference from the installed model itself, so reference and MTP run
    # share one weight grid and greedy ties resolve identically.
    nn.quantize(model, group_size=64, bits=4)
    model.eval()
    assert mlx_fastpath.install(model) is True
    ref = _greedy(model, PROMPT, N_TOKENS)
    eng = _engine(model, _head_for(model))
    text, stats = eng._generate_mtp(PROMPT, N_TOKENS, None, None)
    got = [int(t) for t in text.split()]
    assert stats.draft_proposed > 0, "drafting never ran"
    assert stats.draft_accepted < stats.draft_proposed, (
        "random head should reject: replay arm never exercised")
    assert got == ref, (
        f"diverged at {next(i for i, (a, b) in enumerate(zip(ref, got)) if a != b)}")


def test_zero_model_full_accept_saves_forwards():
    """Drafter and verifier share zeroed weights, so every draft agrees; the
    accept arm plus the drafter catch-up must commit 40 tokens in far fewer
    verify forwards than tokens."""
    model = _build_tiny(zero=True)
    ref = _greedy(model, PROMPT, N_TOKENS)
    head = mlx_mtp.build(model.language_model.args)
    from mlx.utils import tree_map
    head.update(tree_map(mx.zeros_like, head.parameters()))
    head.eval()
    eng = _engine(model, head)
    text, stats = eng._generate_mtp(PROMPT, N_TOKENS, None, None)
    got = [int(t) for t in text.split()]
    assert got == ref
    assert stats.draft_accepted == stats.draft_proposed > 0
    assert stats.forwards < N_TOKENS // 2, (
        f"full accept should batch commits: {stats.forwards} forwards")


def test_adaptive_greedy_is_bit_exact():
    """The cost-model schedule changes WHICH depths run, never what commits:
    greedy output under adaptive drafting must be bit-equal to plain decoding
    with both a garbage head (collapse to skip + probes) and a perfect one
    (ramp toward the cap)."""
    model = _build_tiny()
    ref = _greedy(model, PROMPT, N_TOKENS)
    eng = _engine(model, _head_for(model), adaptive=True)
    text, _ = eng._generate_mtp(PROMPT, N_TOKENS, None, None)
    assert [int(t) for t in text.split()] == ref

    zmodel = _build_tiny(zero=True)
    zref = _greedy(zmodel, PROMPT, N_TOKENS)
    from mlx.utils import tree_map
    zhead = mlx_mtp.build(zmodel.language_model.args)
    zhead.update(tree_map(mx.zeros_like, zhead.parameters()))
    zhead.eval()
    zeng = _engine(zmodel, zhead, adaptive=True)
    ztext, zstats = zeng._generate_mtp(PROMPT, N_TOKENS, None, None)
    assert [int(t) for t in ztext.split()] == zref
    assert zstats.draft_accepted == zstats.draft_proposed


def test_depth_policy_dynamics():
    """Pure-policy contract: hot content ramps to the cap, sustained rejection
    collapses to the free skip but probes its way back, and a near-tie pending
    margin clamps the first positions."""
    from chad.mlx_mtp import DepthPolicy
    hot = DepthPolicy(7, 0.35)
    for _ in range(40):
        d = hot.depth()
        hot.record(d, d, False)
    assert hot.depth() == 7

    cold = DepthPolicy(7, 0.35)
    for _ in range(12):
        d = cold.depth()
        if d:
            cold.record(d, 0, False)
    assert cold.depth() == 0
    # the 16-skip probe must eventually re-offer depth 1
    assert any(cold.depth() == 1 for _ in range(20))

    tie = DepthPolicy(7, 0.35)
    tie.margin = 0.0
    assert tie.depth() <= 1


def test_second_turn_extends_cleanly():
    """The turn leaves _cached_ids exactly matching residency, so an append-only
    second turn must reproduce the fresh-cache reference."""
    model = _build_tiny()
    eng = _engine(model, _head_for(model))
    text1, _ = eng._generate_mtp(PROMPT, 12, None, None)
    got1 = [int(t) for t in text1.split()]
    prompt2 = PROMPT + got1 + [17, 23]
    ref = _greedy(model, prompt2, 12)
    text2, stats2 = eng._generate_mtp(prompt2, 12, None, None)
    got2 = [int(t) for t in text2.split()]
    assert got2 == ref, (
        f"diverged at {next(i for i, (a, b) in enumerate(zip(ref, got2)) if a != b)}")
    assert stats2.cached_tokens > 0, "prefix cache was not reused"


def test_sampled_path_stays_consistent():
    """temp>0: rejection sampling can't be compared token-for-token against a
    reference, but the cache bookkeeping contract is identical. Assert the turn
    completes and residency matches the fed invariant."""
    model = _build_tiny()
    eng = _engine(model, _head_for(model), temp=0.8)
    text, stats = eng._generate_mtp(PROMPT, 25, None, None)
    got = [int(t) for t in text.split()]
    assert len(got) == 25
    assert stats.generated_tokens == 25
    # A budget stop mid-commit can leave accepted-but-unemitted tokens
    # resident (same contract as PLD), so residency is bounded, not exact.
    lo = len(PROMPT) + len(got) - 1
    assert lo <= len(eng._cached_ids) <= lo + eng.mtp_num_draft
