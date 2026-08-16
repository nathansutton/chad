"""Kernel-level tests for the CATS sparse-MLP decode path (mlx_cats).

Small synthetic shapes exercising the same geometry constraints as the 27B
(K % 256 == 0, D % 32 == 0, rows arbitrary). References are stock mx ops.
Not bit-exact vs fp math (quantized kernels); tolerances match the measured
<=0.2% relative error of the prototypes.
"""
import math

import pytest

mx = pytest.importorskip("mlx.core")

from chad import mlx_cats  # noqa: E402

K = 512
I_ROWS = 1024
D = 512


@pytest.fixture(scope="module")
def rig():
    mx.random.seed(7)
    uq, us, ub = mx.quantize(
        mx.random.normal((I_ROWS, K)).astype(mx.float16), group_size=64, bits=3)
    dq, ds, db = mx.quantize(
        mx.random.normal((I_ROWS, D)).astype(mx.float16), group_size=64, bits=4)
    x = (mx.random.normal((K,)) * 0.1).astype(mx.float16)
    g = (mx.random.normal((I_ROWS,)) * 2.0).astype(mx.float16)
    mx.eval(uq, us, ub, dq, ds, db, x, g)
    return dict(uq=uq, us=us, ub=ub, dq=dq, ds=ds, db=db, x=x, g=g)


def _silu_abs(g):
    v = g.astype(mx.float32)
    return mx.abs(v * mx.sigmoid(v))


def test_compact_matches_threshold_set(rig):
    a = _silu_abs(rig["g"])
    thr_v = float(mx.sort(a)[int(I_ROWS * 0.6)])
    idx, cnt = mlx_cats.compact(rig["g"], mx.array([thr_v], dtype=mx.float32),
                                cap=I_ROWS)
    mx.eval(idx, cnt)
    n = int(cnt[0])
    want = {j for j, v in enumerate((a >= thr_v).tolist()) if v}
    assert n == len(want)
    assert set(idx[:n].tolist()) == want


def test_compact_cap_clamps_writes(rig):
    idx, cnt = mlx_cats.compact(rig["g"], mx.array([0.0], dtype=mx.float32),
                                cap=64)
    mx.eval(idx, cnt)
    assert int(cnt[0]) == I_ROWS  # count reports the true total
    assert idx.shape[0] == 64     # writes stayed inside cap
    assert len(set(idx.tolist())) == 64


def test_gather_u_matches_dense(rig):
    a = _silu_abs(rig["g"])
    thr_v = float(mx.sort(a)[int(I_ROWS * 0.7)])
    cap = I_ROWS
    idx, cnt = mlx_cats.compact(rig["g"], mx.array([thr_v], dtype=mx.float32),
                                cap)
    mx.eval(idx, cnt)
    n = int(cnt[0])
    h = mlx_cats.gather_u(rig["x"], rig["g"], rig["uq"], rig["us"], rig["ub"],
                          idx, cnt, cap, K)
    mx.eval(h)
    u_full = mx.quantized_matmul(
        rig["x"].reshape(1, 1, -1), rig["uq"], scales=rig["us"],
        biases=rig["ub"], transpose=True, group_size=64, bits=3)[0, 0]
    t = rig["g"].astype(mx.float32)
    t = t * mx.sigmoid(t)
    ii = idx[:n].astype(mx.int32)
    ref = (t[ii] * u_full[ii].astype(mx.float32)).astype(mx.float16)
    rel = float(mx.abs(h[:n] - ref).max()) / (float(mx.abs(ref).max()) + 1e-9)
    assert rel < 5e-3


def test_row_acc4_matches_dense(rig):
    a = _silu_abs(rig["g"])
    thr_v = float(mx.sort(a)[int(I_ROWS * 0.7)])
    cap = I_ROWS
    idx, cnt = mlx_cats.compact(rig["g"], mx.array([thr_v], dtype=mx.float32),
                                cap)
    mx.eval(idx, cnt)
    n = int(cnt[0])
    h = (mx.random.normal((cap,)) * 0.1).astype(mx.float16)
    mx.eval(h)
    y = mlx_cats.row_acc4(h, rig["dq"], rig["ds"], rig["db"], idx, cnt, cap, D)
    mx.eval(y)
    ii = idx[:n].astype(mx.int32)
    hf = mx.zeros((I_ROWS,), dtype=mx.float32)
    hf[ii] = h[:n].astype(mx.float32)
    wd = mx.dequantize(rig["dq"], rig["ds"], rig["db"], group_size=64,
                       bits=4).astype(mx.float32)
    ref = (hf[None, :] @ wd)[0]
    rel = float(mx.abs(y.astype(mx.float32) - ref).max()) / (
        float(mx.abs(ref).max()) + 1e-9)
    assert rel < 5e-3


def test_sparse_mlp_equals_masked_dense(rig):
    """End-to-end kernel chain == dense MLP with t zeroed outside the set
    (the faithful-sim identity the quality probes rely on)."""
    a = _silu_abs(rig["g"])
    thr_v = float(mx.sort(a)[int(I_ROWS * 0.65)])
    cap = I_ROWS
    idx, cnt = mlx_cats.compact(rig["g"], mx.array([thr_v], dtype=mx.float32),
                                cap)
    h = mlx_cats.gather_u(rig["x"], rig["g"], rig["uq"], rig["us"], rig["ub"],
                          idx, cnt, cap, K)
    y = mlx_cats.row_acc4(h, rig["dq"], rig["ds"], rig["db"], idx, cnt, cap, D)
    mx.eval(y)

    u_full = mx.quantized_matmul(
        rig["x"].reshape(1, 1, -1), rig["uq"], scales=rig["us"],
        biases=rig["ub"], transpose=True, group_size=64, bits=3)[0, 0]
    t = rig["g"].astype(mx.float32)
    t = t * mx.sigmoid(t)
    t = t * (a >= thr_v)
    hd = (t * u_full.astype(mx.float32)).astype(mx.float16)
    wd = mx.dequantize(rig["dq"], rig["ds"], rig["db"], group_size=64,
                       bits=4).astype(mx.float32)
    ref = (hd.astype(mx.float32)[None, :] @ wd)[0]
    rel = float(mx.abs(y.astype(mx.float32) - ref).max()) / (
        float(mx.abs(ref).max()) + 1e-9)
    assert rel < 5e-3


def test_bf16_boundary_cast_semantics(rig):
    """Guard for plan 132 L13b: activations arrive bf16 and the engine casts
    to fp16 at the kernel boundary. The selection truth is defined ON the cast
    values — pin that compact agrees exactly with a python reference computed
    from the same cast tensor."""
    gbf = rig["g"].astype(mx.bfloat16)   # engine-side dtype
    g16 = gbf.astype(mx.float16)         # boundary cast
    a = _silu_abs(g16)
    thr_v = float(mx.sort(a)[int(I_ROWS * 0.6)])
    idx, cnt = mlx_cats.compact(g16, mx.array([thr_v], dtype=mx.float32),
                                I_ROWS)
    mx.eval(idx, cnt)
    n = int(cnt[0])
    want = {j for j, v in enumerate((a >= thr_v).tolist()) if v}
    assert n == len(want)
    assert set(idx[:n].tolist()) == want


def test_enabled_flag(monkeypatch):
    monkeypatch.delenv("CHAD_CATS_R", raising=False)
    assert not mlx_cats.enabled()
    monkeypatch.setenv("CHAD_CATS_R", "0.25")
    assert mlx_cats.enabled()
