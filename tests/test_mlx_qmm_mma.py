"""Small-M MMA quantized matmul: numerics against the stock kernel at every
supported bit width and verify width, and the dispatch gate (only verified
shapes and widths route; decode M=1 never does)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from chad import mlx_qmm_mma as q  # noqa: E402

K, N = 512, 4096


def _weights(bits, seed=0, gs=64):
    mx.random.seed(seed)
    w = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)
    wq, sc, bi = mx.quantize(w, group_size=gs, bits=bits)
    mx.eval(wq, sc, bi)
    return wq, sc, bi


@pytest.mark.parametrize("bits,gs", [(3, 64), (4, 64), (5, 64), (6, 64), (8, 64),
                                     (2, 64), (2, 128), (4, 128)])
def test_matches_stock_kernel_at_every_width(bits, gs):
    wq, sc, bi = _weights(bits, gs=gs)
    for M in range(1, q.M_MAX + 1):
        x = (mx.random.normal((M, K)) * 0.1).astype(mx.bfloat16)
        ref = mx.quantized_matmul(x, wq, scales=sc, biases=bi, transpose=True,
                                  group_size=gs, bits=bits).astype(mx.float32)
        got = q.mma(x, wq, sc, bi, M, N, K, bits, gs).astype(mx.float32)
        rel = float(mx.max(mx.abs(ref - got))) / max(float(mx.max(mx.abs(ref))), 1e-6)
        assert rel < 0.02, (bits, gs, M, rel)


def test_ternary_pack_levels_match_stock():
    """The Prism packs: 2-bit g128 whose codes {0,1,2} decode to {-s, 0, +s} (biases
    == -scales, code 3 unused). The lane's one-word window must cover every code."""
    mx.random.seed(5)
    codes = mx.random.randint(0, 3, (N, K)).astype(mx.uint32)
    wq = mx.zeros((N, K // 16), dtype=mx.uint32)
    for i in range(16):
        wq = wq | (codes[:, i::16] << (2 * i))
    sc = (mx.random.uniform(shape=(N, K // 128)) * 0.02 + 0.005).astype(mx.bfloat16)
    bi = -sc
    mx.eval(wq, sc, bi)
    for M in (1, 4, 8):
        x = (mx.random.normal((M, K)) * 0.1).astype(mx.bfloat16)
        ref = mx.quantized_matmul(x, wq, scales=sc, biases=bi, transpose=True,
                                  group_size=128, bits=2).astype(mx.float32)
        got = q.mma(x, wq, sc, bi, M, N, K, 2, 128).astype(mx.float32)
        rel = float(mx.max(mx.abs(ref - got))) / max(float(mx.max(mx.abs(ref))), 1e-6)
        assert rel < 0.02, (M, rel)


def test_partial_last_tile_is_guarded():
    """N that is not a multiple of the 32-column threadgroup tile: the last tile's
    out-of-range rows must neither read past the weights nor write past out."""
    bits, Np = 3, N + 8 * 3 + 8      # one full tile short, then 3 tiles + 1 row
    mx.random.seed(1)
    w = (mx.random.normal((Np, K)) * 0.02).astype(mx.bfloat16)
    wq, sc, bi = mx.quantize(w, group_size=64, bits=bits)
    for M in (3, 8):
        x = (mx.random.normal((M, K)) * 0.1).astype(mx.bfloat16)
        ref = mx.quantized_matmul(x, wq, scales=sc, biases=bi, transpose=True,
                                  group_size=64, bits=bits).astype(mx.float32)
        got = q.mma(x, wq, sc, bi, M, Np, K, bits).astype(mx.float32)
        assert got.shape == (M, Np)
        rel = float(mx.max(mx.abs(ref - got))) / max(float(mx.max(mx.abs(ref))), 1e-6)
        assert rel < 0.02, (M, rel)


def test_qmm_dispatch_gate():
    wq, sc, bi = _weights(4)
    q.set_wins({(K, N, 4, 64): 3})
    try:
        for M, expect_kernel in ((1, False), (2, False), (3, True), (8, True), (9, False)):
            x = (mx.random.normal((1, M, K)) * 0.1).astype(mx.bfloat16)
            y = q.qmm(x, wq, sc, bi, 64, 4)
            ref = mx.quantized_matmul(x, wq, scales=sc, biases=bi, transpose=True,
                                      group_size=64, bits=4)
            assert y.shape == ref.shape == (1, M, N)
            same = float(mx.max(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32))))
            # stock path is bit-identical to itself; the kernel is only ulp-close
            if not expect_kernel:
                assert same == 0.0, (M, same)
        # an unverified shape / fp16 input always take stock
        x16 = (mx.random.normal((8, K)) * 0.1).astype(mx.float16)
        ref16 = mx.quantized_matmul(x16, wq, scales=sc, biases=bi, transpose=True,
                                    group_size=64, bits=4)
        assert float(mx.max(mx.abs(q.qmm(x16, wq, sc, bi, 64, 4) - ref16))) == 0.0
    finally:
        q.disable()


@pytest.mark.parametrize("bits,gs", [(2, 128), (4, 64)])
def test_tiled_matches_stock_past_one_tile(bits, gs):
    """Widths past M_MAX run as 8-row kernel calls plus a remainder; whatever the
    remainder is (a full tile, a kernel-width one, a stock-width one, a single row)
    every row has to land in its own place."""
    mx.random.seed(2)
    w = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)
    wq, sc, bi = mx.quantize(w, group_size=gs, bits=bits)
    for M in (9, 12, 16, 21):
        x = (mx.random.normal((M, K)) * 0.1).astype(mx.bfloat16)
        ref = mx.quantized_matmul(x, wq, scales=sc, biases=bi, transpose=True,
                                  group_size=gs, bits=bits).astype(mx.float32)
        got = q.tiled(x, wq, sc, bi, M, N, K, bits, gs, 5).astype(mx.float32)
        assert got.shape == (M, N)
        rows = mx.max(mx.abs(ref - got), axis=1) / mx.max(mx.abs(ref))
        assert float(mx.max(rows)) < 0.02, (M, rows.tolist())


def test_qmm_tiles_only_through_the_probed_width():
    wq, sc, bi = _weights(4)
    q.set_wins({(K, N, 4, 64): 3}, tile_max={(K, N, 4, 64): 16})
    try:
        for M, expect_tiled in ((9, True), (16, True), (17, False)):
            x = (mx.random.normal((1, M, K)) * 0.1).astype(mx.bfloat16)
            y = q.qmm(x, wq, sc, bi, 64, 4).astype(mx.float32)
            ref = mx.quantized_matmul(x, wq, scales=sc, biases=bi, transpose=True,
                                      group_size=64, bits=4).astype(mx.float32)
            assert y.shape == (1, M, N)
            # the stock path is bit-identical to itself; the kernel is only ulp-close
            assert (float(mx.max(mx.abs(y - ref))) > 0.0) == expect_tiled, M
            assert float(mx.max(mx.abs(y - ref))) < 0.02 * float(mx.max(mx.abs(ref)))
        saved, tiles = q.wins(), q.tile_max()
        q.disable()
        q.set_wins(saved)                      # the A/B round trip keeps the tiling
        assert q.tile_max() == tiles
    finally:
        q.set_wins(None, tile_max={})


def test_quantized_linear_patch_routes_only_verified_shapes():
    lin = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=4)
    lin2 = nn.QuantizedLinear(K, N // 2, bias=False, group_size=64, bits=4)
    x = (mx.random.normal((1, 8, K)) * 0.1).astype(mx.bfloat16)
    stock = mx.quantized_matmul(x, lin["weight"], scales=lin["scales"],
                                biases=lin["biases"], transpose=True,
                                group_size=64, bits=4)
    stock2 = lin2(x)
    q.set_wins({(K, N, 4, 64): 2})
    try:
        y = lin(x)
        assert y.shape == stock.shape
        rel = float(mx.max(mx.abs(y.astype(mx.float32) - stock.astype(mx.float32)))) \
            / float(mx.max(mx.abs(stock.astype(mx.float32))))
        assert rel < 0.02
        assert float(mx.max(mx.abs(lin2(x) - stock2))) == 0.0   # unverified: stock
    finally:
        q.disable()
    assert float(mx.max(mx.abs(lin(x) - stock))) == 0.0         # disabled: stock


def test_shape_gate():
    assert q.shape_ok(5120, 17408, 3, 64)
    assert q.shape_ok(5120, 248320, 5, 64)
    assert not q.shape_ok(5120, 1024, 4, 64)      # too few columns
    assert not q.shape_ok(5000, 17408, 4, 64)     # K not a multiple of 512
    assert q.shape_ok(5120, 17408, 2, 128)        # the Prism ternary container
    assert not q.shape_ok(5120, 17408, 4, 32)     # group size
    assert not q.shape_ok(5120, 17408, 1, 64)     # unsupported width


def test_eligible_groups_skips_fastpath_placeholders():
    """The decode fastpath leaves (8,)-shaped placeholder arrays in the modules
    whose weights it concatenated; the shape walk must skip them (it used to
    raise, which silently disabled the whole kernel at load)."""
    class _Fused(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=4)
            self.lin["weight"] = mx.zeros((8,), dtype=mx.uint32)   # placeholder
            self.lin["scales"] = mx.zeros((8,), dtype=mx.uint32)
            self.lin["biases"] = mx.zeros((8,), dtype=mx.uint32)
            w = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)
            self._fused_w, self._fused_s, self._fused_b = mx.quantize(
                w, group_size=64, bits=4)
            self._fused_gs, self._fused_bits = 64, 4

    groups = q._eligible_groups(_Fused())
    assert list(groups) == [(K, N, 4, 64)] and len(groups[(K, N, 4, 64)]) == 1
    # a placeholder module must also never be routed by the class patch
    q.set_wins({(K, N, 4, 64): 2})
    try:
        m = _Fused()
        x = (mx.random.normal((1, 8, K)) * 0.1).astype(mx.bfloat16)
        with pytest.raises(Exception):
            m.lin(x)          # stock path on garbage weights raises; the patch fell through
    finally:
        q.disable()
