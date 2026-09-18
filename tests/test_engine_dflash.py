"""DFlash2 block-drafter speculative decoding: distribution safety and the
drafter bookkeeping on the shared speculative loop.

The core promise: whatever the block drafter proposes, the committed output
must be EXACTLY what plain decoding would have produced
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
from test_mlx_fastpath import TINY_DENSE_CFG  # noqa: E402

PROMPT = [5, 9, 11, 22, 33, 44, 55, 66, 77, 88, 101, 102, 7]
N_TOKENS = 40


def _build_tiny(seed=0, zero=False):
    import copy

    from mlx.utils import tree_map
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    cfg = copy.deepcopy(TINY_DENSE_CFG)
    cfg["text_config"]["head_dim"] = 64
    mx.random.seed(seed)
    model = Model(ModelArgs.from_dict(cfg))
    if zero:
        model.update(tree_map(mx.zeros_like, model.parameters()))
    model.eval()
    return model


def _greedy(model, prompt_ids, n):
    cache = cache_utils.make_prompt_cache(model)
    logits = model(mx.array(prompt_ids)[None], cache=cache)
    out = [int(mx.argmax(logits[0, -1]).item())]
    for _ in range(n - 1):
        logits = model(mx.array([out[-1]])[None], cache=cache)
        out.append(int(mx.argmax(logits[0, -1]).item()))
    return out


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
    eng._dflash = drafter
    eng.dflash_num_draft = num_draft
    eng.cache_dir = None
    eng.max_context = None
    eng.effective_ctx = 4096
    eng.model_id = "tiny-hybrid"
    eng._model_path = ""
    eng._cached_ids = []
    eng._rewind_snap = None
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
    text, stats = eng._generate_spec(PROMPT, N_TOKENS, None, None)
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
    text, stats = eng._generate_spec(PROMPT, N_TOKENS, None, None)
    assert _ids(text) == ref
    assert mlx_fastpath.GDN_COLLECTOR is None
    assert mlx_dflash.TAP is None
    assert stats.draft_proposed > 0


def test_no_sliding_window_drafter_is_bit_exact():
    model = _build_tiny()
    ref = _greedy(model, PROMPT, N_TOKENS)
    eng = _engine(model, _drafter_for(model, sliding=False), num_draft=2)
    text, _ = eng._generate_spec(PROMPT, N_TOKENS, None, None)
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
    text, stats = eng._generate_spec(PROMPT, N_TOKENS, None, None)
    assert _ids(text) == ref
    assert stats.draft_accepted == stats.draft_proposed > 0
    # k=3 accepted + 1 bonus per round -> ~N/4 forwards
    assert stats.forwards <= N_TOKENS // 4 + 2


def test_second_turn_extends_cleanly():
    model = _build_tiny()
    eng = _engine(model, _drafter_for(model))
    text1, _ = eng._generate_spec(PROMPT, 12, None, None)
    prompt2 = list(PROMPT) + _ids(text1) + [17, 18, 19]
    ref2 = _greedy(model, prompt2, 12)
    text2, stats2 = eng._generate_spec(prompt2, 12, None, None)
    assert _ids(text2) == ref2
    assert stats2.cached_tokens > 0


def test_sampled_path_stays_consistent():
    model = _build_tiny()
    eng = _engine(model, _drafter_for(model), temp=0.8)
    text, stats = eng._generate_spec(PROMPT, 25, None, None)
    ids = _ids(text)
    assert len(ids) == 25
    assert stats.draft_proposed > 0
    assert 0 <= stats.draft_accepted <= stats.draft_proposed
    assert eng._cached_ids == list(PROMPT[:-1]) + [PROMPT[-1]] + ids[:-1]


def _ledger_matches_cache(eng):
    """Every attention layer holds exactly as many tokens as the ledger records. (The
    recurrent layers carry no position to compare; they move in the same forwards.)"""
    offsets = [c.offset for c in eng._cache if isinstance(c, cache_utils.KVCache)]
    return bool(offsets) and all(o == len(eng._cached_ids) for o in offsets)


def test_raising_callback_keeps_the_turn_cached():
    """A callback that raises mid-turn (a UI bug, say) runs only after the round's tokens
    are recorded, so the loop's ledger is exact. The turn stays cached and the next turn
    extends it bit-exactly, instead of prefilling on top of tokens the ledger forgot."""
    model = _build_tiny()
    eng = _engine(model, _drafter_for(model))
    emitted = []

    def on_token(seg):
        emitted.append(seg)
        if len(emitted) == 9:
            raise RuntimeError("ui died")

    with pytest.raises(RuntimeError, match="ui died"):
        eng._generate_spec(PROMPT, N_TOKENS, on_token, None)
    assert _ledger_matches_cache(eng)
    assert len(eng._cached_ids) > len(PROMPT)          # kept, not dropped
    prompt2 = list(PROMPT) + _ids("".join(emitted)) + [17, 18, 19]
    ref2 = _greedy(model, prompt2, 12)
    text2, stats2 = eng._generate_spec(prompt2, 12, None, None)
    assert _ids(text2) == ref2
    assert stats2.cached_tokens > len(PROMPT)


def test_error_between_forward_and_ledger_drops_the_cache():
    """MLX can raise after a verify forward has moved the cache but before the round's
    tokens are recorded (a Metal error at the accept eval, say). The loop's ledger then
    trails the cache, so publishing it would orphan tokens: the cache must be dropped.
    Raised here from the drafter reconcile, which runs at exactly that point."""
    from chad import engine as eng_mod

    class _DiesAtThirdReconcile(eng_mod._DFlashDrafter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.rounds = 0

        def reconcile(self, *args, **kwargs):
            self.rounds += 1
            if self.rounds == 3:
                raise RuntimeError("metal died")
            return super().reconcile(*args, **kwargs)

    model = _build_tiny()
    eng = _engine(model, _drafter_for(model))
    with pytest.raises(RuntimeError, match="metal died"):
        eng._generate_spec(PROMPT, N_TOKENS, None, None,
                           drafter_cls=_DiesAtThirdReconcile)
    assert eng._cached_ids == []
    assert _ledger_matches_cache(eng)
    text, _ = eng._generate_spec(PROMPT, N_TOKENS, None, None)
    assert _ids(text) == _greedy(model, PROMPT, N_TOKENS)


def test_error_during_prefill_drops_the_cache():
    """A prefill that raises partway has fed chunks the ledger doesn't record, so the
    cache must be dropped. Raised here from the progress hook, which runs at that point:
    a chunk fed, the ledger not yet updated."""
    model = _build_tiny()
    eng = _engine(model, _drafter_for(model))

    def progress(done, total):
        raise RuntimeError("metal died")

    with pytest.raises(RuntimeError, match="metal died"):
        eng._generate_spec(PROMPT, N_TOKENS, None, None, on_prefill_progress=progress)
    assert eng._cached_ids == []
    assert _ledger_matches_cache(eng)
    text, _ = eng._generate_spec(PROMPT, N_TOKENS, None, None)
    assert _ids(text) == _greedy(model, PROMPT, N_TOKENS)


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
    text, stats = eng._generate_spec(PROMPT, N_TOKENS, None, None)
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
    # a cold run: rejections at position 0 collapse the schedule to the free
    # skip, but the 16-skip probe must eventually re-offer a width
    pol = mlx_dflash.block_policy(7)
    for _ in range(30):
        k = pol.depth()
        pol.record(max(k, 1), 0, stopped_early=False)
        pol.observe_cost(k, 0.05 + 0.03 * k)
    assert pol.depth() <= 1
    assert any(pol.depth() == 1 for _ in range(20))
    # a near-tie pending margin clamps the round whatever the streak says
    tie = mlx_dflash.block_policy(7)
    for _ in range(12):
        k = tie.depth()
        tie.record(k, k, stopped_early=False)
    tie.margin = 0.0
    assert tie.depth() <= 1
    # an explicit cap below the block is honored
    assert mlx_dflash.block_policy(2).depth() <= 2



def test_ensure_bundle_completes_a_config_only_download(tmp_path):
    """The measured half-state: mlx-lm's `model*.safetensors` download pattern is
    anchored at the start of the relative path, so `dflash/config.json` arrives (it
    matches `*.json`) and `dflash/model.safetensors` does not. That leaves a bundle
    that looks present and loads nothing, and the base weights look complete so no
    retry ever runs. ensure_bundle heals it; a complete bundle and a local model dir
    must not touch the network."""
    calls = []

    def download(repo, filename):
        calls.append((repo, filename))
        return str(tmp_path / filename)

    mdir = tmp_path / "snapshot"
    bundle = mdir / "dflash"
    bundle.mkdir(parents=True)
    (bundle / "config.json").write_text("{}")

    mlx_dflash.ensure_bundle(str(mdir), "org/model", download=download)
    assert calls == [("org/model", "dflash/model.safetensors")]

    # already complete -> no fetch
    calls.clear()
    (bundle / "model.safetensors").write_bytes(b"x")
    mlx_dflash.ensure_bundle(str(mdir), "org/model", download=download)
    assert calls == []

    # a local model dir is not a repo id -> no fetch, and no bundle at all is not
    # this function's problem (that model simply ships no drafter)
    (bundle / "model.safetensors").unlink()
    mlx_dflash.ensure_bundle(str(mdir), str(tmp_path), download=download)
    mlx_dflash.ensure_bundle(str(tmp_path / "no-bundle"), "org/model", download=download)
    assert calls == []


def test_ensure_bundle_failure_is_soft(tmp_path):
    """Offline or gated: warn and decode without the drafter, never raise into load."""
    attempts = []

    def offline(repo, filename):
        attempts.append((repo, filename))
        raise OSError("offline")

    bundle = tmp_path / "m" / "dflash"
    bundle.mkdir(parents=True)
    (bundle / "config.json").write_text("{}")
    mlx_dflash.ensure_bundle(str(tmp_path / "m"), "org/model", download=offline)  # no raise
    assert attempts == [("org/model", "dflash/model.safetensors")]


def test_sidecar_loads_at_the_width_it_was_built_at(monkeypatch, tmp_path):
    """build_sidecar records bits/group_size in the safetensors header; the loader must
    quantize its skeleton from that, not from load_drafter's default of 4. A mismatch
    raises inside load_drafter's catch-all, which returns None and decodes serially —
    so the assertion that matters is that a drafter comes back at all, and at 8 bits."""
    import json
    import logging

    from mlx.utils import tree_flatten

    model = _build_tiny()
    args = model.language_model.args
    n_layers = int(args.num_hidden_layers)
    hf = {
        "hidden_size": int(args.hidden_size), "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
        "intermediate_size": 128, "vocab_size": int(args.vocab_size),
        "num_target_layers": n_layers, "max_position_embeddings": 512,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "dflash_config": {"block_size": 4, "target_layer_ids": [0, max(1, n_layers - 2)],
                          "mask_token_id": int(args.vocab_size) - 1,
                          "selector_rank": 64, "selector_top_k": 4,
                          "conv_kernel_size": 2, "conv_group_size": 16},
    }
    src = tmp_path / "hf"
    src.mkdir()
    (src / "config.json").write_text(json.dumps(hf))
    mx.random.seed(3)
    bf = mlx_dflash.build(mlx_dflash.DFlashConfig.from_dict(hf))
    mx.save_safetensors(str(src / "model.safetensors"), dict(tree_flatten(bf.parameters())))
    out = mlx_dflash.build_sidecar(str(src), str(tmp_path / "q8"), bits=8, gs=64)

    messages = []

    class _Rec(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    chad_log = logging.getLogger("chad")
    handler, old_level = _Rec(logging.INFO), chad_log.level
    chad_log.addHandler(handler)
    chad_log.setLevel(logging.INFO)
    monkeypatch.setenv("CHAD_DFLASH_PATH", out)
    try:
        drafter = mlx_dflash.load_drafter(model, str(tmp_path / "weights"))
    finally:
        chad_log.removeHandler(handler)
        chad_log.setLevel(old_level)

    assert drafter is not None, messages
    assert drafter.fc.bits == 8 and drafter.fc.group_size == 64
    assert any("8-bit g64" in m for m in messages), messages


def test_schedule_is_the_default_width_policy():
    """The per-round verified-width schedule is ON by default, chosen on the floor: a
    fixed full-block round costs ~2.2 serial steps whatever it commits, and on
    low-acceptance text it measured below serial decoding where the schedule held
    above it (benchmarks/spec_decode.py). CHAD_DFLASH_ADAPTIVE=0 / CHAD_DFLASH_DRAFT=N
    select the fixed arm at load."""
    assert Engine.dflash_adaptive is True
    assert Engine.dflash_num_draft == 7


def _tiny_sidecar(model, tmp_path, bits=4):
    """A quantized DFlash sidecar for the tiny target, built the way the CLI builds one."""
    import json

    from mlx.utils import tree_flatten

    args = model.language_model.args
    n_layers = int(args.num_hidden_layers)
    hf = {
        "hidden_size": int(args.hidden_size), "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
        "intermediate_size": 128, "vocab_size": int(args.vocab_size),
        "num_target_layers": n_layers, "max_position_embeddings": 512,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "dflash_config": {"block_size": 4, "target_layer_ids": [0, max(1, n_layers - 2)],
                          "mask_token_id": int(args.vocab_size) - 1,
                          "selector_rank": 64, "selector_top_k": 4,
                          "conv_kernel_size": 2, "conv_group_size": 16},
    }
    src = tmp_path / "hf"
    src.mkdir()
    (src / "config.json").write_text(json.dumps(hf))
    mx.random.seed(3)
    bf = mlx_dflash.build(mlx_dflash.DFlashConfig.from_dict(hf))
    mx.save_safetensors(str(src / "model.safetensors"), dict(tree_flatten(bf.parameters())))
    return mlx_dflash.build_sidecar(str(src), str(tmp_path / "sidecar"), bits=bits, gs=64)


def test_drafter_borrows_a_donor_bundle_for_a_shape_match(monkeypatch, tmp_path):
    """A checkpoint with no bundled drafter (the Prism ternary pack) takes the sidecar a
    DONORS sibling of the same (hidden, layers, vocab) bundles — resolved file by file
    through the hub cache — and nothing else: a shape with no donor, or a model that
    is itself the donor, decodes serially."""
    monkeypatch.delenv("CHAD_DFLASH_PATH", raising=False)
    model = _build_tiny()
    out = _tiny_sidecar(model, tmp_path)
    key = mlx_dflash._target_key(model)
    asked = []

    def donor_file(repo_id, filename):
        asked.append((repo_id, filename))
        return os.path.join(out, os.path.basename(filename))

    weights = tmp_path / "weights"          # no dflash/ inside
    weights.mkdir()

    assert mlx_dflash.load_drafter(model, str(weights), donors={}, fetch=donor_file) is None
    assert asked == []

    donors = {key: "sibling/repo"}
    assert mlx_dflash.load_drafter(model, str(weights), repo_id="sibling/repo",
                                   donors=donors, fetch=donor_file) is None
    assert asked == []                        # the donor itself: nothing to borrow

    drafter = mlx_dflash.load_drafter(model, str(weights), repo_id="other/pack",
                                      donors=donors, fetch=donor_file)
    assert drafter is not None
    assert asked == [("sibling/repo", "dflash/config.json"),
                     ("sibling/repo", "dflash/model.safetensors")]


def test_donor_bundle_split_across_snapshots_is_healed_or_refused(tmp_path):
    """The donor's config and weights resolve one file at a time, so a stale cache hit
    on the config beside weights downloaded into the current revision is two snapshot
    dirs. Returning the config's dir handed the loader a bundle with no weights, and the
    catch-all turned that into silent serial decoding. The weights' snapshot is the live
    one: the config is re-fetched into it, and a split that cannot be healed is refused
    rather than returned."""
    import shutil

    old, new = tmp_path / "snap-old" / "dflash", tmp_path / "snap-new" / "dflash"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / "config.json").write_text("{}")
    (new / "model.safetensors").write_bytes(b"")
    calls = []

    def healing(repo_id, filename):
        calls.append(filename)
        if filename.endswith("model.safetensors"):
            return str(new / "model.safetensors")
        if calls.count(filename) > 1:             # the forced re-fetch
            shutil.copy(old / "config.json", new / "config.json")
            return str(new / "config.json")
        return str(old / "config.json")

    donors = {"k": "sibling/repo"}
    assert mlx_dflash.donor_bundle_dir("k", "other/pack", donors, healing) == str(new)
    assert calls == ["dflash/config.json", "dflash/model.safetensors",
                     "dflash/config.json"]

    os.remove(new / "config.json")

    def stuck(repo_id, filename):
        name = os.path.basename(filename)
        return str((new if name == "model.safetensors" else old) / name)

    assert mlx_dflash.donor_bundle_dir("k", "other/pack", donors, stuck) is None


def test_round_cost_seed_is_per_weight_width():
    """The ladder's shape belongs to the checkpoint: the engine seeds the schedule with
    the one measured at the target's weight width, and the schedule still spans skip to
    full block on it."""
    assert mlx_dflash.round_costs(2) == mlx_dflash.BLOCK_ROUND_COSTS_2BIT
    assert mlx_dflash.round_costs(3) == mlx_dflash.round_costs(None) \
        == mlx_dflash.BLOCK_ROUND_COSTS
    cold = mlx_dflash.block_policy(7, mlx_dflash.round_costs(2))
    for _ in range(12):
        cold.record(7, 0, stopped_early=False)
    assert cold.depth() == 0
    hot = mlx_dflash.block_policy(7, mlx_dflash.round_costs(2))
    for _ in range(12):
        hot.record(7, 7, stopped_early=False)
    assert hot.depth() == 7


def test_learned_seed_carries_the_measured_ladder_in_seed_units():
    """The next turn's seed is this turn's, moved halfway to what was measured at every
    depth measured enough, in SEED units: a turn at twice the wall-clock scale (a longer
    context) must learn the same ladder, and a depth seen once must not move at all."""
    seed = list(mlx_dflash.BLOCK_ROUND_COSTS)
    assert mlx_dflash.block_policy(7).learned_seed() == seed          # nothing observed

    def learned(scale):
        pol = mlx_dflash.block_policy(7)
        for _ in range(8):
            pol.observe_cost(0, 0.050 * scale)          # the anchor: most observed
        for _ in range(4):
            pol.observe_cost(3, 0.150 * scale)          # 3.0 steps, seed says 2.30
        pol.observe_cost(5, 9.0 * scale)                # one inflated sample
        return pol.learned_seed()

    a, b = learned(1.0), learned(2.0)
    assert a == pytest.approx(b)
    assert a[3] == pytest.approx(0.5 * seed[3] + 0.5 * 3.0)
    assert a[5] == seed[5] and a[0] == pytest.approx(seed[0])
    assert mlx_dflash.block_policy(7, a).cost(3) == pytest.approx(a[3])
