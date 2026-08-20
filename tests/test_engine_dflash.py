"""DFlash2 block-drafter speculative decoding: distribution safety and the
drafter bookkeeping on the shared speculative loop.

Same promise as the MTP suite: whatever the block drafter proposes, the
committed output must be EXACTLY what plain decoding would have produced
(greedy: bit-equal tokens). A random-weight drafter proposes garbage, so it
exercises the full-rollback arm (GDN replay + KV trim) on every round and the
drafter-context bookkeeping across rejections; an oracle drafter that knows
the true continuation exercises the full-accept arm and the forward savings.

These run on the tiny synthetic qwen3_5 DENSE hybrid from the fastpath suite —
the shipped model's geometry — with an unquantized tiny drafter so exact
argmax comparisons are meaningful."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models import cache as cache_utils  # noqa: E402

from chad import mlx_dflash, mlx_fastpath  # noqa: E402
from chad.engine import Engine  # noqa: E402
from test_engine_mtp import PROMPT, _build_tiny, _greedy, _Tok  # noqa: E402

N_TOKENS = 40


def _drafter_for(model, seed=7, block_size=4, sliding=True):
    """A tiny random-weight DFlash2 drafter shaped for the tiny target."""
    args = model.language_model.args
    n_layers = int(args.num_hidden_layers)
    taps = (0, max(1, n_layers - 2))
    cfg = mlx_dflash.DFlashConfig(
        hidden_size=int(args.hidden_size), num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        intermediate_size=64, vocab_size=int(args.vocab_size),
        rms_norm_eps=1e-6, rope_theta=1e4, max_position_embeddings=512,
        block_size=block_size, target_layer_ids=taps, num_target_layers=n_layers,
        mask_token_id=int(args.vocab_size) - 1,
        layer_types=(("sliding_attention", "full_attention") if sliding
                     else ("full_attention", "full_attention")),
        sliding_window=8 if sliding else None,
        selector_rank=8, selector_top_k=4, conv_kernel_size=2, conv_group_size=16)
    mx.random.seed(seed)
    d = mlx_dflash.build(cfg)
    d.eval()
    mx.eval(d.parameters())
    lm = model.language_model
    embed = lm.model.embed_tokens
    d.bind(embed, embed.as_linear if lm.args.tie_word_embeddings else lm.lm_head)
    assert mlx_dflash.install_tap(model, cfg.target_layer_ids)
    return d


def _engine(model, drafter, temp=0.0, num_draft=3, adaptive=False):
    # adaptive=False by default: these tests pin the fixed-width contract
    # (forwards-count assertions assume a constant k), and the synthetic
    # models emit flat logits whose zero top-2 margin legitimately clamps
    # the adaptive schedule shallow.
    eng = object.__new__(Engine)
    eng.dflash_adaptive = adaptive
    eng.model = model
    eng.tok = _Tok()
    eng.kv_bits = None
    eng.temp = temp
    eng.min_p = eng.top_p = 0.0
    eng.prompt_lookup = False
    eng.enable_pld_hybrid = False
    eng._mtp_head = None
    eng._dflash = drafter
    eng.dflash_num_draft = num_draft
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


def _ids(text):
    return [int(t) for t in text.split()]


def test_random_drafter_greedy_is_bit_exact():
    model = _build_tiny()
    ref = _greedy(model, PROMPT, N_TOKENS)
    eng = _engine(model, _drafter_for(model))
    text, stats = eng._generate_spec("dflash", PROMPT, N_TOKENS, None, None)
    assert _ids(text) == ref
    assert stats.draft_proposed > 0
    # garbage drafts: (almost) nothing accepted, so ~one forward per token
    assert stats.forwards >= N_TOKENS // 2


def test_random_drafter_greedy_bit_exact_through_capture_replay():
    """With the fastpath installed the rollback is the GDN capture-replay
    primitive (no re-feed); the drafter's tap rides the same forwards."""
    import mlx.nn as nn
    model = _build_tiny()
    # install() engages only on quantized hybrids; quantize THEN take the
    # reference from the installed model itself, so reference and spec run
    # share one weight grid and greedy ties resolve identically.
    nn.quantize(model, group_size=64, bits=4)
    model.eval()
    assert mlx_fastpath.install(model) is True
    ref = _greedy(model, PROMPT, N_TOKENS)
    eng = _engine(model, _drafter_for(model))
    assert mlx_fastpath.GDN_COLLECTOR is None
    text, stats = eng._generate_spec("dflash", PROMPT, N_TOKENS, None, None)
    assert _ids(text) == ref
    assert mlx_fastpath.GDN_COLLECTOR is None
    assert mlx_dflash.TAP is None
    assert stats.draft_proposed > 0


def test_no_sliding_window_drafter_is_bit_exact():
    model = _build_tiny()
    ref = _greedy(model, PROMPT, N_TOKENS)
    eng = _engine(model, _drafter_for(model, sliding=False), num_draft=2)
    text, _ = eng._generate_spec("dflash", PROMPT, N_TOKENS, None, None)
    assert _ids(text) == ref


class _Oracle:
    """A drafter that knows the true continuation: every draft is accepted.
    Exercises the full-accept arm and the pending-context bookkeeping."""

    def __init__(self, real, truth, prompt_len):
        self.config = real.config
        self._real = real
        self._truth = truth
        self._pos = 0          # committed generated tokens so far

    def make_cache(self):
        return self._real.make_cache()

    def project_ctx(self, fused):
        return self._real.project_ctx(fused)

    def append_ctx(self, rows, cache):
        self._real.append_ctx(rows, cache)

    def select_block(self, block, fused, cache, *, cap, anchor_id, uniforms=None,
                     temperature=1.0):
        if fused is not None:
            self._real.append_ctx(self._real.project_ctx(fused), cache)
        ids = mx.array(self._truth[self._pos : self._pos + cap], dtype=mx.int32)
        self._pos += cap + 1   # all accepted + the bonus token
        return ids, None, None


def test_oracle_drafter_full_accept_saves_forwards():
    model = _build_tiny()
    ref = _greedy(model, PROMPT, N_TOKENS)
    real = _drafter_for(model)
    eng = _engine(model, _Oracle(real, ref, len(PROMPT)), num_draft=3)
    text, stats = eng._generate_spec("dflash", PROMPT, N_TOKENS, None, None)
    assert _ids(text) == ref
    assert stats.draft_accepted == stats.draft_proposed > 0
    # k=3 accepted + 1 bonus per round -> ~N/4 forwards
    assert stats.forwards <= N_TOKENS // 4 + 2


def test_second_turn_extends_cleanly():
    model = _build_tiny()
    eng = _engine(model, _drafter_for(model))
    text1, _ = eng._generate_spec("dflash", PROMPT, 12, None, None)
    prompt2 = list(PROMPT) + _ids(text1) + [17, 18, 19]
    ref2 = _greedy(model, prompt2, 12)
    text2, stats2 = eng._generate_spec("dflash", prompt2, 12, None, None)
    assert _ids(text2) == ref2
    assert stats2.cached_tokens > 0


def test_sampled_path_stays_consistent():
    model = _build_tiny()
    eng = _engine(model, _drafter_for(model), temp=0.8)
    text, stats = eng._generate_spec("dflash", PROMPT, 25, None, None)
    ids = _ids(text)
    assert len(ids) == 25
    assert stats.draft_proposed > 0
    assert 0 <= stats.draft_accepted <= stats.draft_proposed
    assert eng._cached_ids == list(PROMPT[:-1]) + [PROMPT[-1]] + ids[:-1]


def test_tap_captures_layer_outputs_without_changing_the_forward():
    model = _build_tiny()
    lm = model.language_model
    ids = mx.array(PROMPT, dtype=mx.uint32)[None]
    c1 = cache_utils.make_prompt_cache(model)
    ref = lm.model(ids, cache=c1)
    taps = (0, int(lm.args.num_hidden_layers) - 1)
    assert mlx_dflash.install_tap(model, taps)
    sink = {}
    mlx_dflash.TAP = sink
    try:
        c2 = cache_utils.make_prompt_cache(model)
        got = lm.model(ids, cache=c2)
    finally:
        mlx_dflash.TAP = None
    assert float(mx.abs(ref - got).max()) == 0.0
    assert set(sink) == set(taps)
    h = int(lm.args.hidden_size)
    for i in taps:
        assert sink[i].shape == (1, len(PROMPT), h)
    # unarmed: the tapped layers publish nothing
    c3 = cache_utils.make_prompt_cache(model)
    lm.model(ids, cache=c3)
    assert set(sink) == set(taps)


def test_config_from_hf_dict():
    cfg = mlx_dflash.DFlashConfig.from_dict({
        "architectures": ["DFlash2DraftModel"],
        "dflash_config": {"block_size": 8, "conv_group_size": 16,
                          "conv_kernel_size": 2, "mask_token_id": 248070,
                          "selector_rank": 256, "selector_top_k": 16,
                          "target_layer_ids": [5, 19, 33, 47, 61]},
        "head_dim": 128, "hidden_size": 5120, "intermediate_size": 17408,
        "layer_types": ["sliding_attention"] * 5, "max_position_embeddings": 262144,
        "num_attention_heads": 32, "num_hidden_layers": 5, "num_key_value_heads": 8,
        "num_target_layers": 64, "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        "sliding_window": 2048, "use_sliding_window": True, "vocab_size": 248320})
    assert cfg.block_size == 8 and cfg.selector_top_k == 16
    assert cfg.target_layer_ids == (5, 19, 33, 47, 61)
    assert cfg.rope_theta == 1e7 and cfg.rope_scaling is None
    assert cfg.sliding_window == 2048 and all(
        t == "sliding_attention" for t in cfg.layer_types)


def test_adaptive_width_greedy_is_bit_exact():
    model = _build_tiny()
    ref = _greedy(model, PROMPT, N_TOKENS)
    eng = _engine(model, _drafter_for(model), num_draft=3, adaptive=True)
    text, stats = eng._generate_spec("dflash", PROMPT, N_TOKENS, None, None)
    assert _ids(text) == ref
    assert stats.forwards >= 1


def test_block_policy_dynamics():
    pol = mlx_dflash.block_policy(7)
    # optimistic prior: drafts from the first round, never past the block cap
    assert 1 <= pol.depth() <= 7
    # a hot run: full accepts climb to the full block (no plateau gate)
    for _ in range(12):
        k = pol.depth()
        pol.record(k, k, stopped_early=False)
        pol.observe_cost(k, 0.05 + 0.03 * k)
    assert pol.depth() == 7
    # a cold run: rejections at position 0 collapse the schedule
    pol = mlx_dflash.block_policy(7)
    for _ in range(30):
        k = pol.depth()
        pol.record(max(k, 1), 0, stopped_early=False)
        pol.observe_cost(k, 0.05 + 0.03 * k)
    assert pol.depth() <= 1
    # an explicit cap below the block is honored
    assert mlx_dflash.block_policy(2).depth() <= 2

