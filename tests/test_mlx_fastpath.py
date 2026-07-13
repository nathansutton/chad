"""mlx_fastpath: the decode fast-path must be a pure re-plumbing of the same
computation. These tests build a tiny synthetic qwen3_5_moe hybrid (2 layers:
one GDN + one attention, 8 experts top-2, 4-bit) — no downloaded weights — and
assert that installing the fast-path changes neither greedy token choices nor
(beyond fp-rounding noise) the logits, on both the compiled S==1 decode path
and the stock-graph S>1 prefill path that shares the fused weights.

The class-level patches must also leave a NON-installed model of the same
classes behaving stock (foreign-instance fallback guards)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.nn as nn  # noqa: E402
from mlx_lm.models import cache as cache_utils  # noqa: E402
from mlx_lm.models.qwen3_5 import Model, ModelArgs  # noqa: E402

from chad import mlx_fastpath  # noqa: E402

TINY_CFG = {
    "model_type": "qwen3_5_moe",
    "text_config": {
        "model_type": "qwen3_5_moe",
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "full_attention_interval": 2,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 4,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 64,
        "vocab_size": 256,
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


def build_tiny():
    mx.random.seed(0)
    model = Model(ModelArgs.from_dict(TINY_CFG))
    nn.quantize(model, group_size=64, bits=4)
    model.eval()
    return model


def greedy(model, prompt_ids, n=6):
    cache = cache_utils.make_prompt_cache(model)
    logits = model(mx.array(prompt_ids, dtype=mx.uint32)[None], cache=cache)
    mx.eval([c.state for c in cache])
    prefill_logits = logits[0, -1].astype(mx.float32)
    cur = mx.argmax(prefill_logits)
    mx.eval(cur, prefill_logits)
    out, step_logits = [int(cur)], []
    for _ in range(n):
        logits = model(mx.array([out[-1]], dtype=mx.uint32)[None], cache=cache)
        l0 = logits[0, -1].astype(mx.float32)
        t = mx.argmax(l0)
        mx.eval(t, l0)
        step_logits.append(l0)
        out.append(int(t))
    return out, prefill_logits, step_logits


PROMPT = [3, 141, 59, 26, 53, 58, 97, 93, 23, 84, 62, 64, 33, 83, 27, 95]


def test_fastpath_greedy_equivalence():
    model = build_tiny()
    ref_ids, ref_pre, ref_steps = greedy(model, PROMPT)

    assert mlx_fastpath.install(model) is True
    got_ids, got_pre, got_steps = greedy(model, PROMPT)

    # prefill (S>1, stock graph + fused weights): row-math is unchanged, but
    # matmul tiling may round differently — allow only tiny drift.
    pre_err = float(mx.abs(ref_pre - got_pre).max())
    assert pre_err < 5e-2, f"prefill logits drifted {pre_err}"
    # decode: token choices must match exactly; logits within rounding noise.
    assert got_ids == ref_ids
    worst = max(float(mx.abs(a - b).max()) for a, b in zip(ref_steps, got_steps))
    assert worst < 5e-2, f"decode logits drifted {worst}"


def test_foreign_instance_falls_back_to_stock():
    """After install() patched the classes, a second NON-installed model of the
    same classes must still produce its own stock outputs (hasattr guards)."""
    patched = build_tiny()
    virgin = build_tiny()  # same seed -> same weights
    ref_ids, ref_pre, _ = greedy(virgin, PROMPT)

    assert mlx_fastpath.install(patched) is True
    got_ids, got_pre, _ = greedy(virgin, PROMPT)  # virgin NOT installed
    assert got_ids == ref_ids
    assert float(mx.abs(ref_pre - got_pre).max()) == 0.0


def test_flag_disables_install(monkeypatch):
    monkeypatch.setenv("CHAD_NO_FASTPATH", "1")
    model = build_tiny()
    assert mlx_fastpath.install(model) is False


def test_wrong_architecture_is_noop():
    class NotAModel:
        pass

    assert mlx_fastpath.install(NotAModel()) is False
