"""mlx_moe_fused: the fused MoE decode kernels must be a re-plumbing of the
same computation. Builds a synthetic qwen3_5_moe hybrid at the REAL per-layer
geometry the kernels bake in (hidden 2048, expert inter 512, top-8, expert
gate|up 2-bit + down 3-bit / shared 6-bit / router+seg 8-bit, gs 64) but with only 16 experts and
2 layers — the kernels never index by expert count, so this exercises every
code path without the 13 GB checkpoint.

Asserts: greedy parity through the full model (decode uses the kernels,
prefill the bit-exact patched stock graph); batched decode falls back to the
compiled body; foreign instances stay stock; the flag disables the stage; and
the tiny 4-bit fastpath model never engages it."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.nn as nn  # noqa: E402
from mlx_lm.models.qwen3_5 import Model, ModelArgs  # noqa: E402

from chad import mlx_fastpath  # noqa: E402
from test_mlx_fastpath import TINY_CFG, build_tiny, greedy  # noqa: E402

REAL_CFG = {
    "model_type": "qwen3_5_moe",
    "text_config": {
        "model_type": "qwen3_5_moe",
        "hidden_size": 2048,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "full_attention_interval": 2,
        "linear_num_value_heads": 32,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 16,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "vocab_size": 512,
        "tie_word_embeddings": False,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
        },
    },
}


def _bits_for(path):
    if "switch_mlp" in path:
        return 3 if "down_proj" in path else 2
    if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
        return 8
    return 6


def build_real_geometry():
    mx.random.seed(0)
    model = Model(ModelArgs.from_dict(REAL_CFG))

    def predicate(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        return {"group_size": 64, "bits": _bits_for(path)}

    nn.quantize(model, group_size=64, bits=6, class_predicate=predicate)
    model.set_dtype(mx.bfloat16)
    model.eval()
    return model


PROMPT = [3, 141, 59, 26, 53, 58, 97, 93, 23, 84, 62, 64, 33, 83, 27, 95]


def _moe_layers(model):
    return model.language_model.model.layers


def assert_greedy_parity(ref_ids, ref_pre, ref_steps, got_ids, got_steps, env):
    """Margin-aware greedy comparison for random-weight models: logits are
    compared only while the generated prefixes agree, and a token flip is
    acceptable only on a near-tie (reference margin within the rounding
    envelope). A flip on a decisive margin is a real bug."""
    div = next((i for i, (r, g) in enumerate(zip(ref_ids, got_ids)) if r != g),
               len(ref_ids))
    comparable = ref_steps[:max(div - 1, 0)]
    if comparable:
        worst = max(float(mx.abs(a - b).max())
                    for a, b in zip(comparable, got_steps))
        assert worst < env, f"decode logits drifted {worst}"
    if div < len(ref_ids):
        flip_logits = ref_pre if div == 0 else ref_steps[div - 1]
        margin = float(flip_logits[ref_ids[div]] - flip_logits[got_ids[div]])
        assert margin < 2 * env, \
            f"token flip at index {div} on decisive margin {margin}"



def test_fused_engages_and_greedy_matches():
    model = build_real_geometry()
    ref_ids, ref_pre, ref_steps = greedy(model, PROMPT)

    assert mlx_fastpath.install(model) is True
    for layer in _moe_layers(model):
        assert hasattr(layer.mlp, "_rt_w"), "router|seg concat missing"
        assert hasattr(layer.mlp, "_sh_w"), "shared gate|up concat missing"
        assert layer.mlp._fused_step is not None, "kernels did not build"

    got_ids, got_pre, got_steps = greedy(model, PROMPT)
    # prefill: bit-exact re-plumb (row slices of fused qmv), tiling noise only
    pre_err = float(mx.abs(ref_pre - got_pre).max())
    assert pre_err < 5e-2, f"prefill logits drifted {pre_err}"
    # decode: kernels are bf16-accumulation class; token-identity on
    # meaningful logits is gated on the real checkpoint.
    assert_greedy_parity(ref_ids, ref_pre, ref_steps, got_ids, got_steps,
                         env=2.5e-1)


def test_batched_decode_falls_back_to_body():
    model = build_real_geometry()
    assert mlx_fastpath.install(model) is True
    layer = _moe_layers(model)[0]
    h = mx.random.normal((2, 1, 2048)).astype(mx.bfloat16)
    got = layer._moe_fast(h)
    ref = layer._moe_body_fast(h)
    mx.eval(got, ref)
    assert got.shape == (2, 1, 2048)
    assert float(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max()) == 0.0


def test_kernel_step_matches_body():
    """The two custom dispatches vs the compiled stock-graph body on the same
    fused weights: bf16-class agreement, identical expert routing by design."""
    model = build_real_geometry()
    assert mlx_fastpath.install(model) is True
    layer = _moe_layers(model)[0]
    h = mx.random.normal((1, 1, 2048)).astype(mx.bfloat16)
    a = layer.mlp._fused_step(h)
    b = layer._moe_body_fast(h)
    mx.eval(a, b)
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    rel = float(mx.abs(a32 - b32).max() / (mx.abs(b32).max() + 1e-9))
    assert rel < 3e-2, f"fused step drifted {rel} rel from body"


def test_foreign_instance_falls_back_to_stock():
    patched = build_real_geometry()
    virgin = build_real_geometry()  # same seed -> same weights
    ref_ids, ref_pre, _ = greedy(virgin, PROMPT)

    assert mlx_fastpath.install(patched) is True
    got_ids, got_pre, _ = greedy(virgin, PROMPT)  # virgin NOT installed
    assert got_ids == ref_ids
    assert float(mx.abs(ref_pre - got_pre).max()) == 0.0


def test_flag_disables_fused_stage(monkeypatch):
    monkeypatch.setenv("CHAD_NO_MOE_FUSED", "1")
    model = build_real_geometry()
    assert mlx_fastpath.install(model) is True  # fastpath itself still on
    for layer in _moe_layers(model):
        assert not hasattr(layer.mlp, "_rt_w")
        assert getattr(layer.mlp, "_fused_step", None) is None


def test_tiny_fastpath_model_does_not_engage():
    """The 128-hidden 4-bit fastpath test model must never match the gate."""
    model = build_tiny()
    assert mlx_fastpath.install(model) is True
    for layer in model.language_model.model.layers:
        assert not hasattr(layer.mlp, "_rt_w")
    # and TINY_CFG really is the config this guards against
    assert TINY_CFG["text_config"]["hidden_size"] != 2048


def test_kernel_build_failure_falls_back_to_body(monkeypatch):
    """kernels=None must still leave a CORRECT model: transforms applied,
    _fused_step None, and decode running the compiled stock-graph body over
    the fused router|seg + shared weights (the moek branch of _moe_body) —
    greedy-parity against a virgin stock model."""
    from chad import mlx_moe_fused

    virgin = build_real_geometry()
    ref_ids, ref_pre, ref_steps = greedy(virgin, PROMPT)

    monkeypatch.setattr(mlx_moe_fused, "_build_kernels", lambda model: None)
    model = build_real_geometry()  # same seed -> same weights
    assert mlx_fastpath.install(model) is True
    for layer in _moe_layers(model):
        assert hasattr(layer.mlp, "_rt_w"), "transforms must still apply"
        assert layer.mlp._fused_step is None
        # wrapper not installed: _moe_fast IS the body, no split
        assert not hasattr(layer, "_moe_body_fast")

    got_ids, got_pre, got_steps = greedy(model, PROMPT)
    pre_err = float(mx.abs(ref_pre - got_pre).max())
    assert pre_err < 5e-2, f"prefill logits drifted {pre_err}"
    # decode: op-graph class drift only (no custom-kernel accumulation in
    # this path) — tighter envelope than the kernel test.
    assert_greedy_parity(ref_ids, ref_pre, ref_steps, got_ids, got_steps,
                         env=5e-2)


def test_geometry_gate_rejects_each_deviation():
    """_matches must reject every single-parameter deviation from the exact
    35B geometry (checked on layer 1, so EVERY layer is inspected)."""
    from chad import mlx_moe_fused

    model = build_real_geometry()
    mlx_fastpath._concat_expert_gate_up(model)  # _matches runs after this
    assert mlx_moe_fused._matches(model) is True
    assert mlx_moe_fused._matches(object()) is False  # no layers at all

    mlp = _moe_layers(model)[1].mlp
    sw = mlp.switch_mlp

    def rejected_after(mutate, restore):
        mutate()
        try:
            assert mlx_moe_fused._matches(model) is False
        finally:
            restore()
        assert mlx_moe_fused._matches(model) is True, "restore failed"

    # routing config
    old = mlp.top_k
    rejected_after(lambda: setattr(mlp, "top_k", 4),
                   lambda: setattr(mlp, "top_k", old))
    rejected_after(lambda: setattr(mlp, "norm_topk_prob", False),
                   lambda: setattr(mlp, "norm_topk_prob", True))
    olde = mlp.num_experts
    rejected_after(lambda: setattr(mlp, "num_experts", 8),  # E <= top_k
                   lambda: setattr(mlp, "num_experts", olde))
    rejected_after(lambda: setattr(mlp, "sharding_group", object()),
                   lambda: setattr(mlp, "sharding_group", None))
    # expert gate|up bits and shape
    rejected_after(lambda: setattr(sw, "_fused_bits", 4),
                   lambda: setattr(sw, "_fused_bits", 2))
    oldw = sw._fused_w
    rejected_after(lambda: setattr(sw, "_fused_w",
                                   oldw[..., : oldw.shape[-1] // 2]),
                   lambda: setattr(sw, "_fused_w", oldw))
    # expert down / shared expert / router quant recipes
    rejected_after(lambda: setattr(sw.down_proj, "bits", 4),
                   lambda: setattr(sw.down_proj, "bits", 3))
    rejected_after(lambda: setattr(mlp.shared_expert.gate_proj, "bits", 4),
                   lambda: setattr(mlp.shared_expert.gate_proj, "bits", 6))
    rejected_after(lambda: setattr(mlp.gate, "group_size", 32),
                   lambda: setattr(mlp.gate, "group_size", 64))
    # per-layer dtype mismatch between fused experts and router scales
    olds = mlp.gate.scales
    rejected_after(lambda: setattr(mlp.gate, "scales",
                                   olds.astype(mx.float32)),
                   lambda: setattr(mlp.gate, "scales", olds))
    # global dtype gate: fp32 scales everywhere on layer 0 must be rejected
    mlp0 = _moe_layers(model)[0].mlp
    olds0, oldf0 = mlp0.gate.scales, mlp0.switch_mlp._fused_s

    def _to_fp32():
        mlp0.gate.scales = olds0.astype(mx.float32)
        mlp0.switch_mlp._fused_s = oldf0.astype(mx.float32)

    def _back():
        mlp0.gate.scales = olds0
        mlp0.switch_mlp._fused_s = oldf0

    rejected_after(_to_fp32, _back)


def test_missing_mlx_returns_false(monkeypatch):
    """install() must be a clean no-op when mlx.core cannot be imported."""
    from chad import mlx_moe_fused

    monkeypatch.setitem(sys.modules, "mlx.core", None)  # import -> ImportError
    assert mlx_moe_fused.install(object()) is False


def test_install_failure_is_contained(monkeypatch):
    """An unexpected error inside the stage must be swallowed (return False),
    never propagate into model loading."""
    from chad import mlx_moe_fused

    model = build_real_geometry()
    mlx_fastpath._concat_expert_gate_up(model)

    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(mlx_moe_fused, "_patch_block", boom)
    assert mlx_moe_fused.install(model) is False


def test_mid_loop_transform_failure_leaves_correct_mixed_model(monkeypatch):
    """A transform failing on layer N>0 must leave a MIXED but correct model:
    transformed layers served by the class patch, untouched layers by the
    stock path — greedy parity against a virgin model either way. This is the
    install() atomicity invariant (patch installed before the loop, per-layer
    attrs assigned only after every fused array exists)."""
    from chad import mlx_moe_fused

    virgin = build_real_geometry()
    ref_ids, ref_pre, ref_steps = greedy(virgin, PROMPT)

    real = mlx_moe_fused._transform_layer
    calls = {"n": 0}

    def flaky(mlp):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated OOM")
        real(mlp)

    monkeypatch.setattr(mlx_moe_fused, "_transform_layer", flaky)
    model = build_real_geometry()  # same seed -> same weights
    assert mlx_fastpath.install(model) is True  # fastpath itself survives
    layers = _moe_layers(model)
    assert hasattr(layers[0].mlp, "_rt_w"), "layer 0 should be transformed"
    assert not hasattr(layers[1].mlp, "_rt_w"), "layer 1 should be stock"

    got_ids, got_pre, got_steps = greedy(model, PROMPT)
    pre_err = float(mx.abs(ref_pre - got_pre).max())
    assert pre_err < 5e-2, f"prefill logits drifted {pre_err}"
    assert_greedy_parity(ref_ids, ref_pre, ref_steps, got_ids, got_steps,
                         env=5e-2)


def test_repeated_install_does_not_stack_patches():
    """Every class-level patch must be idempotent: a second install() (second
    engine in one process) must not re-wrap any patched __call__."""
    from mlx_lm.models import qwen3_5 as q35
    from mlx_lm.models import qwen3_next as q3n
    from mlx_lm.models import switch_layers as sl

    m1 = build_real_geometry()
    assert mlx_fastpath.install(m1) is True
    firsts = (q3n.Qwen3NextSparseMoeBlock.__call__, sl.SwitchGLU.__call__,
              q35.GatedDeltaNet.__call__, q35.DecoderLayer.__call__)
    assert getattr(firsts[0], "_chad_moe_fused", False)
    m2 = build_real_geometry()
    assert mlx_fastpath.install(m2) is True
    assert (q3n.Qwen3NextSparseMoeBlock.__call__, sl.SwitchGLU.__call__,
            q35.GatedDeltaNet.__call__, q35.DecoderLayer.__call__) == firsts
