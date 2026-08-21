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

# Kernel-output tests only make sense where the kernel is numerically sound. On
# M1-class GPUs (GitHub's macos-14 arm64 runners) the fused kernel silently returns
# nan on small-n partial-chunk shapes — install() now refuses it there via the same
# self-check, so decode falls back to stock and these tests skip rather than assert
# against a kernel production chad would never use on that hardware.
requires_healthy_kernel = pytest.mark.skipif(
    not mlx_qsdpa.kernel_healthy(),
    reason="qsdpa kernel fails its numeric self-check on this GPU/toolchain "
           "(e.g. M1-class CI runners); install() refuses it here")

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


@requires_healthy_kernel
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


@requires_healthy_kernel
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


def _make_wide(hq, hkv, n, s, dtype, seed=5):
    mx.random.seed(seed)
    q = mx.random.normal((B, hq, s, D)).astype(dtype)
    k = (0.7 * mx.random.normal((B, hkv, n, D))).astype(dtype)
    v = (0.7 * mx.random.normal((B, hkv, n, D))).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


def _reference_wide(q, cache, n):
    """Per-row tail-causal reference at arbitrary hq (the module-level
    _reference is pinned to HQ=16)."""
    kd = mx.dequantize(*cache.keys, group_size=64, bits=8)[..., :n, :]
    vd = mx.dequantize(*cache.values, group_size=64, bits=8)[..., :n, :]
    hq, s = q.shape[1], q.shape[2]
    hkv = kd.shape[1]
    qf = (q.astype(mx.float32) * SCALE).reshape(B, hkv, hq // hkv, s, D)
    scores = qf @ mx.expand_dims(kd.astype(mx.float32), 2).swapaxes(-1, -2)
    qi = mx.arange(n - s, n)[:, None]
    ki = mx.arange(n)[None]
    scores = mx.where(qi >= ki, scores, -mx.inf)
    p = mx.softmax(scores, axis=-1, precise=True)
    return (p @ mx.expand_dims(vd.astype(mx.float32), 2)).reshape(B, hq, s, D)


@requires_healthy_kernel
@pytest.mark.parametrize("dispatch", ["per_row", "wide_kernel"])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("hq,hkv", [(24, 4), (16, 2), (16, 4)])  # gqa 6/8/4
@pytest.mark.parametrize("n", [9, 100, 1024, 5000])
@pytest.mark.parametrize("s", [2, 3, 4])
def test_wide_matches_causal_reference(dispatch, dtype, hq, hkv, n, s,
                                       monkeypatch):
    """S>1 is the speculative verify forward's attention: both dispatches (the
    default per-row S=1 calls and the CHAD_QSDPA_WIDE_KERNEL one-read
    kernel) must match the per-row tail-causal dequantize->fp32 reference,
    including the partial-chunk edges (n=9 with S=4 leaves rows with
    sub-chunk limits) that the S=1 kernel's history says are where silent
    poison lives."""
    if dispatch == "wide_kernel":
        monkeypatch.setenv("CHAD_QSDPA_WIDE_KERNEL", "1")
    else:
        monkeypatch.delenv("CHAD_QSDPA_WIDE_KERNEL", raising=False)
    # Evaluate the cap UNDER this dispatch's env: only the wide kernel carries
    # the GQA*S threadgroup bound, and at gqa 8 that bound is 4.
    if s > mlx_qsdpa._wide_s_max(hq // hkv, n):
        pytest.skip("S over this dispatch's cap at this gqa")
    q, k, v = _make_wide(hq, hkv, n, s, dtype)
    cache = _fill_cache(k, v, dtype)
    out = mlx_qsdpa.qsdpa(q, cache.keys, cache.values, SCALE, n)
    ref = _reference_wide(q, cache, n)
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    tol = 2e-3 if dtype == mx.float16 else 8e-3
    assert err < tol, f"{dispatch} hq={hq} n={n} s={s} err={err}"


@requires_healthy_kernel
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("s", [7, 10, 12, 16, 20, 24])
def test_deep_widths_match_causal_reference(dtype, s):
    """The plateau widths the adaptive schedule jumps to. These only reach the
    kernel above _SGM_MIN_N (the SGM retile), and they were unreachable — and
    so untested — while the cap sat at 6. The RT-split form serves all of them:
    its threadgroup allocation is S-independent, and the row-tile count rides
    in grid.y, which is exactly the property the old cap failed to model."""
    n = mlx_qsdpa._SGM_MIN_N + 522          # off a chunk boundary on purpose
    q, k, v = _make_wide(24, 4, n, s, dtype)
    cache = _fill_cache(k, v, dtype)
    assert mlx_qsdpa._eligible(q, cache, "causal")
    out = mlx_qsdpa.qsdpa(q, cache.keys, cache.values, SCALE, n)
    err = mx.abs(out.astype(mx.float32) - _reference_wide(q, cache, n)).max().item()
    tol = 2e-3 if dtype == mx.float16 else 8e-3
    assert err < tol, f"s={s} {dtype} err={err}"


@requires_healthy_kernel
def test_wide_kernel_ignores_padded_tail():
    n, s = 300, 3
    q, k, v = _make_wide(24, 4, n, s, mx.float16)
    cache = _fill_cache(k, v, mx.float16)
    out1 = mlx_qsdpa.qsdpa(q, cache.keys, cache.values, SCALE, n)
    kw, ks, kb = cache.keys
    kw[..., n:, :] = mx.full(kw[..., n:, :].shape, 0xFFFFFFFF, dtype=mx.uint32)
    ks[..., n:, :] = mx.full(ks[..., n:, :].shape, 100.0, dtype=ks.dtype)
    mx.eval(kw, ks)
    out2 = mlx_qsdpa.qsdpa(q, (kw, ks, kb), cache.values, SCALE, n)
    assert mx.array_equal(out1, out2).item()


def test_wide_eligibility_gates(monkeypatch):
    """S>1 requires the causal mask string, an S within the threadgroup
    budget for its gqa, and enough cache; CHAD_NO_QSDPA_WIDE opts out without
    touching the S==1 path."""
    n = 64
    q, k, v = _make_wide(24, 4, n, 3, mx.float16)
    cache = _fill_cache(k, v, mx.float16)
    assert mlx_qsdpa._eligible(q, cache, "causal")
    assert not mlx_qsdpa._eligible(q, cache, None)          # full-tail attention: not ours
    # The cap that applies is the one for the schedule this context will
    # dispatch — n=64 is below _SGM_MIN_N, so that is the per-row cap.
    smax = mlx_qsdpa._wide_s_max(6, n)
    assert smax == mlx_qsdpa._PER_ROW_S_CAP
    q_big, *_ = _make_wide(24, 4, n, smax + 1, mx.float16)
    assert not mlx_qsdpa._eligible(q_big, cache, "causal")  # past the crossover
    monkeypatch.setenv("CHAD_NO_QSDPA_WIDE", "1")
    assert not mlx_qsdpa._eligible(q, cache, "causal")
    q1, *_ = _make_wide(24, 4, n, 1, mx.float16)
    assert mlx_qsdpa._eligible(q1, cache, None)             # S==1 path untouched
    monkeypatch.delenv("CHAD_NO_QSDPA_WIDE")


def test_wide_s_max_is_per_schedule(monkeypatch):
    """The S cap belongs to whichever schedule `qsdpa` will dispatch, not to
    the one kernel that has a threadgroup bound.

    Only `_kernel_wide` stages GQA*S q rows in threadgroup memory, and it runs
    only under CHAD_QSDPA_WIDE_KERNEL. Applying its 32 KB bound to every path
    capped the default schedule at 6 and dropped every DepthPolicy plateau
    width onto the dequantize fallback — measured up to 30% slower per verify
    round at 38k context."""
    big_n = mlx_qsdpa._SGM_MIN_N

    # Opt-in wide kernel: the budget bound still applies, and still bites.
    monkeypatch.setenv("CHAD_QSDPA_WIDE_KERNEL", "1")
    for gqa in (4, 6, 8):
        smax = mlx_qsdpa._wide_s_max(gqa, big_n)
        ch = max(8 // gqa, 1) * gqa
        assert (4 * ch * 256 + gqa * smax * 256) * 2 <= 32768
        assert (4 * ch * 256 + gqa * (smax + 1) * 256) * 2 > 32768 or smax == 6
        assert smax >= 2   # every supported tier can serve at least k=1 verify
    monkeypatch.delenv("CHAD_QSDPA_WIDE_KERNEL")

    # Default: the SGM retile's threadgroup use is S-independent, so the cap is
    # the measured crossover against the dequantize fallback — not a budget.
    for gqa in (4, 6, 8):
        assert mlx_qsdpa._wide_s_max(gqa, big_n) == mlx_qsdpa._WIDE_S_CAP
    assert mlx_qsdpa._WIDE_S_CAP == 24

    # Below the SGM floor, and with the retile disabled for bisection, S>1 is
    # served by replaying the S==1 kernel — linear in S, so it crosses over
    # sooner and takes the lower cap.
    assert mlx_qsdpa._wide_s_max(6, big_n - 1) == mlx_qsdpa._PER_ROW_S_CAP
    monkeypatch.setenv("CHAD_NO_QSDPA_WIDE_SGM", "1")
    assert mlx_qsdpa._wide_s_max(6, big_n) == mlx_qsdpa._PER_ROW_S_CAP
    monkeypatch.delenv("CHAD_NO_QSDPA_WIDE_SGM")
    assert mlx_qsdpa._PER_ROW_S_CAP < mlx_qsdpa._WIDE_S_CAP


def test_dispatchable_verify_widths_are_eligible():
    """Regression guard for the cap bug: every verify width chad can dispatch
    must land on the fused path at a realistic context, except the topmost —
    S=32 is measured slower fused than dequantized, and the cap is set to
    exclude exactly it."""
    dflash = pytest.importorskip("chad.mlx_dflash")
    n = 4096                                    # a mid-session context
    q, k, v = _make_wide(24, 4, n, 1, mx.float16)
    cache = _fill_cache(k, v, mx.float16)
    # the block schedule's widths (2..8) plus wide-PLD's ladder up to 32
    widths = sorted({d + 1 for d in dflash.WidthPolicy.CANDIDATES if d > 0}
                    | {10, 12, 16, 20, 24, 32})
    fused = []
    for s in widths:
        qs, *_ = _make_wide(24, 4, n, s, mx.float16)
        if mlx_qsdpa._eligible(qs, cache, "causal"):
            fused.append(s)
    assert fused == [w for w in widths if w <= 24], fused
    assert 8 in fused and 24 in fused       # the full block and the wide ladder
    assert 32 not in fused                  # the top rung deliberately is not


def test_pick_blocks_table():
    """S==1: below 16k every gqa tier keeps 32 (measured neutral-or-better);
    at n >= 16384 the split widens to the measured winner, 128 at gqa 4 and 64
    otherwise — including the shipped gqa 6, re-measured in that branch.
    Always a multiple of pass 2's BN=32."""
    for gqa in (4, 6, 8):
        for n in (100, 8192, 16000):
            assert mlx_qsdpa._pick_blocks(n, gqa=gqa) == 32
    for n in (16384, 32000, 98304):
        assert mlx_qsdpa._pick_blocks(n, gqa=8) == 64
        assert mlx_qsdpa._pick_blocks(n, gqa=6) == 64
        assert mlx_qsdpa._pick_blocks(n, gqa=4) == 128
    for n in (100, 16384, 99999):
        for gqa in (4, 6, 8):
            assert mlx_qsdpa._pick_blocks(n, gqa) % 32 == 0


def test_pick_blocks_narrows_only_for_the_rt_split_kernel(monkeypatch):
    """The narrow split tracks the KERNEL, not the width. RT-split hands pass 2
    `blocks * 2` entries per head, so its fp32 partials slab is twice as deep
    per block as any other schedule's and S times as tall; the one-read form at
    S<=4 (and the per-row replay at S=2) want the same wide split single-token
    decode wants. Measured: b32 loses 1.6-5.7% at S<=4 and wins 1.1-6.1% at
    S>=5, in both contexts — the crossover sits exactly on the kernel change."""
    monkeypatch.delenv("CHAD_QSDPA_WIDE_SGM_RT", raising=False)
    monkeypatch.delenv("CHAD_NO_QSDPA_WIDE_SGM", raising=False)
    monkeypatch.delenv("CHAD_QSDPA_WIDE_KERNEL", raising=False)
    wide_n = 32768
    for gqa in (4, 6, 8):
        one_read = mlx_qsdpa._pick_blocks(wide_n, gqa, 1)
        for s in (1, 2, 3, 4):                  # S==1 table applies
            assert mlx_qsdpa._pick_blocks(wide_n, gqa, s) == one_read
        for s in (5, 6, 10, 16, 24):            # RT-split: narrow
            assert mlx_qsdpa._pick_blocks(wide_n, gqa, s) == 32
    # Below 16k the table is already 32, so the rule is invisible there.
    for s in (1, 4, 24):
        assert mlx_qsdpa._pick_blocks(8192, 6, s) == 32
    # Forcing RT-split at a narrow width moves the split with it...
    monkeypatch.setenv("CHAD_QSDPA_WIDE_SGM_RT", "1")
    assert mlx_qsdpa._pick_blocks(wide_n, 6, 3) == 32
    monkeypatch.delenv("CHAD_QSDPA_WIDE_SGM_RT")
    # ...and a schedule that is not RT-split at all keeps the table, however
    # wide the step, because no kernel with the deep slab ever runs.
    monkeypatch.setenv("CHAD_NO_QSDPA_WIDE_SGM", "1")
    assert mlx_qsdpa._pick_blocks(wide_n, 6, 16) == mlx_qsdpa._pick_blocks(wide_n, 6, 1)
    monkeypatch.delenv("CHAD_NO_QSDPA_WIDE_SGM")
    # Below the SGM floor no retile runs either, at any width.
    assert mlx_qsdpa._pick_blocks(mlx_qsdpa._SGM_MIN_N - 1, 6, 16) == 32
    # Always a multiple of pass 2's BN, on every branch.
    for s in (1, 3, 5, 24):
        for n in (100, 16384, 99999):
            assert mlx_qsdpa._pick_blocks(n, 6, s) % 32 == 0


def test_dispatch_predicates_match_the_branch(monkeypatch):
    """_pick_blocks sizes a slab for a kernel it does not choose, so its view of
    which kernel runs has to be the same one `qsdpa` branches on. Both read
    these two helpers; this pins their contract."""
    monkeypatch.delenv("CHAD_QSDPA_WIDE_SGM_RT", raising=False)
    monkeypatch.delenv("CHAD_NO_QSDPA_WIDE_SGM", raising=False)
    monkeypatch.delenv("CHAD_QSDPA_WIDE_KERNEL", raising=False)
    big = mlx_qsdpa._SGM_MIN_N
    assert not mlx_qsdpa._sgm_wide_ok(big, 2)        # per-row
    assert mlx_qsdpa._sgm_wide_ok(big, 3)
    assert not mlx_qsdpa._sgm_wide_ok(big - 1, 3)    # below the floor
    assert not mlx_qsdpa._uses_rt_split(big, 4)      # one-read tops out at 4
    assert mlx_qsdpa._uses_rt_split(big, 5)
    for flag in ("CHAD_NO_QSDPA_WIDE_SGM", "CHAD_QSDPA_WIDE_KERNEL"):
        monkeypatch.setenv(flag, "1")
        assert not mlx_qsdpa._sgm_wide_ok(big, 8)
        assert not mlx_qsdpa._uses_rt_split(big, 8)
        monkeypatch.delenv(flag)


@requires_healthy_kernel
def test_gqa4_wide_split_matches_reference():
    """The gqa-4 blocks=128 tier (n >= 16384) is a partials-slab shape no other
    test reaches; the per-head kernel must still match the fp32 reference there."""
    n = 16500
    q, k, v = _make(n, mx.float16, hkv=4)
    cache = _fill_cache(k, v, mx.float16)
    assert mlx_qsdpa._pick_blocks(n, 4) == 128  # the tier under test
    out = mlx_qsdpa.qsdpa(q, cache.keys, cache.values, SCALE, n)
    ref = _reference(q, cache, n)
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    assert err < 2e-3, err


@requires_healthy_kernel
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


@requires_healthy_kernel
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


def test_self_check_gate_catches_poisoned_kernel(monkeypatch):
    """A kernel that silently returns nan (the M1-runner failure mode) must fail
    kernel_healthy() and make install() refuse — the runtime try/except cannot see
    it (nothing raises), so this gate is the only line of defense."""
    def nan_kernel(q, keys, values, scale, n):
        return mx.full(q.shape, float("nan"), dtype=q.dtype)

    monkeypatch.setattr(mlx_qsdpa, "qsdpa", nan_kernel)
    monkeypatch.setattr(mlx_qsdpa, "_KERNEL_HEALTHY", None)  # drop the cached verdict
    assert mlx_qsdpa.kernel_healthy() is False

    # A fresh (unpatched) seam + broken kernel -> install refuses. install() returns
    # True early when the seam already carries _chad_qsdpa, and an earlier test in
    # this process may have installed it for real — so restore a clean seam rather
    # than skipping. Skipping made the assertion below silently not run in a
    # whole-suite pass, which is where it matters most.
    from mlx_lm.models import base as lm_base

    def unpatched(queries, keys, values, cache, scale, mask=None, sinks=None):
        raise AssertionError("stand-in seam should never be called")

    monkeypatch.setattr(lm_base, "scaled_dot_product_attention", unpatched)
    assert not getattr(lm_base.scaled_dot_product_attention, "_chad_qsdpa", False)
    assert mlx_qsdpa.install() is False
    # ...and it refused BEFORE touching the seam, so nothing is left half-patched.
    assert lm_base.scaled_dot_product_attention is unpatched


def test_self_check_result_is_cached(monkeypatch):
    calls = []
    real = mlx_qsdpa.qsdpa

    def counting(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(mlx_qsdpa, "qsdpa", counting)
    monkeypatch.setattr(mlx_qsdpa, "_KERNEL_HEALTHY", None)
    first = mlx_qsdpa.kernel_healthy()
    n_after_first = len(calls)
    assert mlx_qsdpa.kernel_healthy() is first
    assert len(calls) == n_after_first  # second call answered from the cache


@requires_healthy_kernel
@pytest.mark.parametrize("n", [2048, 4095, 5000, 8192, 12345, 16500])
def test_sgm_retile_agrees_with_per_head_kernel(n):
    """The simdgroup_matrix retile (n >= _SGM_MIN_N, gqa 8) and the per-head
    kernel are two schedules of identical math, so they must agree far more
    tightly than either agrees with the fp32 reference. Covers ragged tails
    (n % 8 != 0), n small enough to leave virtual blocks empty, and one n in
    the blocks=64 tier (>= 16384) that real long-context decode runs in."""
    q, k, v = _make(n, mx.bfloat16, hkv=2)
    cache = _fill_cache(k, v, mx.bfloat16)
    kq = (cache.keys[0], cache.keys[1], cache.keys[2])
    vq = (cache.values[0], cache.values[1], cache.values[2])

    assert n >= mlx_qsdpa._SGM_MIN_N          # this n must select the retile
    retiled = mlx_qsdpa.qsdpa(q, kq, vq, SCALE, n)
    monkey = mlx_qsdpa._SGM_MIN_N
    try:
        mlx_qsdpa._SGM_MIN_N = 1 << 30        # force the per-head kernel
        per_head = mlx_qsdpa.qsdpa(q, kq, vq, SCALE, n)
    finally:
        mlx_qsdpa._SGM_MIN_N = monkey
    mx.eval(retiled, per_head)

    err = mx.abs(retiled.astype(mx.float32)
                 - per_head.astype(mx.float32)).max().item()
    assert err < 6e-3, f"n={n} retile vs per-head err={err}"


@requires_healthy_kernel
def test_sgm_retile_declines_gqa4():
    """gqa==4 (the 9B) must keep the per-head kernel: the retile's 8x8 score
    tile has no 8th head row to fill and would read past Qsh."""
    n = 5000
    q, k, v = _make(n, mx.bfloat16, hkv=4)
    cache = _fill_cache(k, v, mx.bfloat16)
    out = mlx_qsdpa.qsdpa(q, (cache.keys[0], cache.keys[1], cache.keys[2]),
                          (cache.values[0], cache.values[1], cache.values[2]),
                          SCALE, n)
    ref = _reference(q, cache, n)
    mx.eval(out, ref)
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    assert err == err, "gqa4 produced nan (retile must not engage here)"
    assert err < 8e-3, f"gqa4 err={err}"


@requires_healthy_kernel
def test_warm_widths_builds_only_the_templated_variants():
    """Pass 1 is templated on the verify width, so warm_widths exists to build
    those variants off the critical path. Widths under 3 dispatch the per-head
    kernel the self-check already built, so they are not variants and must not
    be counted."""
    warmed = mlx_qsdpa.warm_widths([1, 2, 3, 5, 9, 32], hq=24, hkv=4,
                                   dtype=mx.float16)
    assert warmed == 4, f"expected S in 3,5,9,32 warmed, got {warmed}"
    # idempotent and deduped: a repeat run rebuilds nothing new but still
    # reports what the configuration asked for
    assert mlx_qsdpa.warm_widths([5, 5, 5], hq=24, hkv=4,
                                 dtype=mx.float16) == 1


@requires_healthy_kernel
def test_warm_widths_matches_reference_after_warming():
    """Warming must be a pure compile: the same width run afterwards still has
    to agree with the dequantize->fp32 reference."""
    n, s = 2570, 9
    mlx_qsdpa.warm_widths([s], hq=24, hkv=4, dtype=mx.float16)
    q, k, v = _make(n, mx.float16, hkv=4, s=s)
    cache = _fill_cache(k, v, mx.float16)
    out = mlx_qsdpa.qsdpa(q, cache.keys, cache.values, SCALE, n)
    ref = _reference(q, cache, n, mask="causal")
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    assert err < 4e-3, f"S={s} after warm-up err={err}"


def test_warm_widths_declines_uncovered_shapes():
    """A shape the kernel does not serve, or a width list with nothing
    templated in it, warms nothing — and never raises into load()."""
    assert mlx_qsdpa.warm_widths([4], hq=24, hkv=5, dtype=mx.float16) == 0
    assert mlx_qsdpa.warm_widths([4], hq=24, hkv=0, dtype=mx.float16) == 0
    assert mlx_qsdpa.warm_widths([1, 2], hq=24, hkv=4, dtype=mx.float16) == 0
    assert mlx_qsdpa.warm_widths([], hq=24, hkv=4, dtype=mx.float16) == 0
