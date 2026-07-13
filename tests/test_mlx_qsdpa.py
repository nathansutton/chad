"""mlx_qsdpa: the fused quantized-KV decode attention must match the
dequantize->fp32 reference on the exact shape it patches (D=256, GQA=8,
8-bit group-64), decline everything else, and slot into mlx_lm's
scaled_dot_product_attention seam without changing non-quantized calls.
No downloaded weights — synthetic q/K/V only."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.cache import KVCache, QuantizedKVCache  # noqa: E402

from chad import mlx_qsdpa  # noqa: E402

B, HQ, D = 1, 16, 256
HKV = 2          # 35B shape (gqa 8); the 9B is HKV=4 (gqa 4)
SCALE = D ** -0.5


def _make(n, dtype, seed=3, hkv=HKV, s=1):
    mx.random.seed(seed)
    q = mx.random.normal((B, HQ, s, D)).astype(dtype)
    k = (0.7 * mx.random.normal((B, hkv, n, D))).astype(dtype)
    v = (0.7 * mx.random.normal((B, hkv, n, D))).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


def _fill_cache(k, v, dtype):
    """Build a QuantizedKVCache holding k/v (padded allocation included)."""
    c = QuantizedKVCache(group_size=64, bits=8)
    c.update_and_fetch(k, v)
    return c


def _reference(q, cache, n, mask=None):
    kd = mx.dequantize(*cache.keys, group_size=64, bits=8)[..., :n, :]
    vd = mx.dequantize(*cache.values, group_size=64, bits=8)[..., :n, :]
    hkv, s = kd.shape[1], q.shape[2]
    qf = (q.astype(mx.float32) * SCALE).reshape(B, hkv, HQ // hkv, s, D)
    kf = mx.expand_dims(kd.astype(mx.float32), 2)
    vf = mx.expand_dims(vd.astype(mx.float32), 2)
    scores = qf @ kf.swapaxes(-1, -2)
    if mask == "causal":
        qi = mx.arange(n - s, n)[:, None]
        ki = mx.arange(n)[None]
        scores = mx.where(qi >= ki, scores, -mx.inf)
    p = mx.softmax(scores, axis=-1, precise=True)
    return (p @ vf).reshape(B, HQ, s, D)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("hkv", [2, 4])
@pytest.mark.parametrize("n", [3, 100, 1024, 5000])
def test_kernel_matches_dequant_reference(dtype, hkv, n):
    q, k, v = _make(n, dtype, hkv=hkv)
    cache = _fill_cache(k, v, dtype)
    out = mlx_qsdpa.qsdpa(q, cache.keys, cache.values, SCALE, n)
    ref = _reference(q, cache, n)
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    tol = 2e-3 if dtype == mx.float16 else 8e-3
    assert err < tol, f"n={n} hkv={hkv} err={err}"


def test_kernel_ignores_padded_tail():
    """Positions past `n` in the padded buffers must not affect the output."""
    n = 300  # step=256 pads to 512
    q, k, v = _make(n, mx.float16)
    cache = _fill_cache(k, v, mx.float16)
    out1 = mlx_qsdpa.qsdpa(q, cache.keys, cache.values, SCALE, n)
    # poison the tail
    kw, ks, kb = cache.keys
    kw[..., n:, :] = mx.full(kw[..., n:, :].shape, 0xFFFFFFFF, dtype=mx.uint32)
    ks[..., n:, :] = mx.full(ks[..., n:, :].shape, 100.0, dtype=ks.dtype)
    mx.eval(kw, ks)
    out2 = mlx_qsdpa.qsdpa(q, (kw, ks, kb), cache.values, SCALE, n)
    assert mx.array_equal(out1, out2).item()


def test_eligibility_gates():
    q, k, v = _make(64, mx.float16)
    cache = _fill_cache(k, v, mx.float16)
    assert mlx_qsdpa._eligible(q, cache, None)
    assert mlx_qsdpa._eligible(q, cache, "causal")
    # restricting masks, wrong shapes, wrong quant params all decline
    assert not mlx_qsdpa._eligible(q, cache, mx.zeros((1, 1, 1, 64), dtype=mx.bool_))
    assert not mlx_qsdpa._eligible(q.astype(mx.float32), cache, None)
    assert not mlx_qsdpa._eligible(mx.concatenate([q, q], axis=2), cache, None)
    assert not mlx_qsdpa._eligible(q[:, :, :, :128], cache, None)
    assert mlx_qsdpa._eligible(q[:, :8], cache, None)      # gqa 4: covered
    assert not mlx_qsdpa._eligible(q[:, :4], cache, None)  # gqa 2: not
    c4 = QuantizedKVCache(group_size=64, bits=4)
    c4.update_and_fetch(k, v)
    assert not mlx_qsdpa._eligible(q, c4, None)
    empty = QuantizedKVCache(group_size=64, bits=8)
    assert not mlx_qsdpa._eligible(q, empty, None)


def test_install_patches_seam_and_matches():
    from mlx_lm.models import base as lm_base

    assert mlx_qsdpa.install()
    assert getattr(lm_base.scaled_dot_product_attention, "_chad_qsdpa", False)
    assert mlx_qsdpa.install()  # idempotent

    n = 700
    q, k, v = _make(n, mx.float16)
    cache = _fill_cache(k, v, mx.float16)
    keys = tuple(x[..., :n, :] for x in cache.keys)
    values = tuple(x[..., :n, :] for x in cache.values)
    out = lm_base.scaled_dot_product_attention(
        q, keys, values, cache=cache, scale=SCALE, mask=None)
    ref = _reference(q, cache, n)
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    assert err < 2e-3

    # the qwen3_next module-level import must have been rebound too
    from mlx_lm.models import qwen3_next
    assert getattr(qwen3_next.scaled_dot_product_attention, "_chad_qsdpa", False)


def test_covers():
    assert mlx_qsdpa.covers(256, 8)      # 35B
    assert mlx_qsdpa.covers(256, 4)      # 9B
    assert not mlx_qsdpa.covers(128, 8)
    assert not mlx_qsdpa.covers(256, 2)


def test_prefill_dequant_path_matches():
    """S>1 over a quantized cache must route to dequant + fused sdpa and
    match the dequantized-fp32 causal reference."""
    from mlx_lm.models import base as lm_base

    mlx_qsdpa.install()
    n_prev, s_new = 512, 64
    q, k, v = _make(n_prev + s_new, mx.float16, s=s_new)
    cache = _fill_cache(k, v, mx.float16)
    n = n_prev + s_new
    keys = tuple(x[..., :n, :] for x in cache.keys)
    values = tuple(x[..., :n, :] for x in cache.values)
    out = lm_base.scaled_dot_product_attention(
        q, keys, values, cache=cache, scale=SCALE, mask="causal")
    ref = _reference(q, cache, n, mask="causal")
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    assert err < 2e-3, err


def test_install_leaves_fp16_path_stock():
    from mlx_lm.models import base as lm_base

    mlx_qsdpa.install()
    n = 128
    q, k, v = _make(n, mx.float16)
    cache = KVCache()
    keys, values = cache.update_and_fetch(k, v)
    out = lm_base.scaled_dot_product_attention(
        q, keys, values, cache=cache, scale=SCALE, mask=None)
    ref = mx.fast.scaled_dot_product_attention(q, keys, values, scale=SCALE)
    assert mx.array_equal(out, ref).item()


def test_no_qsdpa_flag_blocks_install(monkeypatch):
    monkeypatch.setenv("CHAD_NO_QSDPA", "1")
    assert mlx_qsdpa.install() is False
