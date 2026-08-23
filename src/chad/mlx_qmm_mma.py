"""Small-M quantized matmul for speculative verify: an MMA kernel that reads each
weight group ONCE for all rows.

``mx.quantized_matmul`` is at roofline for M=1 (decode) and tiles well from M≈13
up (prefill), but in between — exactly where speculative verification lives, M =
draft width + 1 — its cost grows nearly linearly in M: the GEMV path re-pays the
weight read per row. Measured on this engine's verify ladder: ~33 ms per extra
row on the 3-bit 27B (a 56 ms serial step), which is why the DFlash block's width
was pinned at 4 when its acceptance would carry 7.

Lineage: avlp12's ``qmm_mma4`` (github.com/avlp12/mlx-lm ``fast_qmm.py``, MIT) as
vendored in ARahim3/mlx-dspark (``small_m_qmm.py``, MIT, Copyright (c) 2026
ARahim3) — an 8x8 ``simdgroup_matrix`` tile covers M <= 8 exactly, so each weight
group is dequantized once and reused by every row, with K split across the 8
simdgroups of a threadgroup. That design staged the dequantized tile through
threadgroup memory (two barriers per chunk) and measured ~68 GB/s against the
stock GEMV's 156. The kernel here keeps the tiling idea and rebuilds the loop:

* Both MMA operands are written straight into the ``simdgroup_matrix`` fragment
  registers (``thread_elements()``: lane holds row (qid&4)+((lane>>1)&3), columns
  (qid&2)*2+(lane&1)*2 and +1 — the layout MLX's own steel GEMM relies on). No
  threadgroup staging, no barriers inside the K loop; threadgroup memory only
  holds the final split-K reduction.
* The product is computed transposed, out^T = W x^T, and k inside each 64-group
  is permuted as k = c*8 + kt identically on both operands, so every lane's
  weight values are 16 CONTIGUOUS values of one row (two 64-bit windows, all
  extraction shifts compile-time) and its x values are two contiguous 16-byte
  loads — and rows of x past M are simply zero-filled in-kernel, so M < 8 needs
  no padded copy of x.
* Weights enter the MMA as the exact bf16 of (2^bits + v), built by a bit-insert
  into a constant (no int->float->bf16 conversion per value); each 64-group is
  accumulated in fp32 and folded in once: acc += s*C_g + (b - 2^bits*s)*sum(x_g),
  with the per-group row sums of x computed in-kernel from the fragment (three
  xor-shuffles). The dequantized weight is never rounded to bf16, so this is
  CLOSER to the stock kernel than the staged version was (max rel err ~0.003 vs
  ~0.007 on the shipped shapes). 8-bit, whose 2^8+v is not exact in bf16, uses
  the exact bf16 of v instead.
* Each simdgroup carries four 8-column tiles (32 output columns per threadgroup),
  which amortizes the x-side work; eight tiles starve occupancy on N=5120.

Why it stops here: measured on this chip the simdgroup MMA issues at ~5.4 TFLOPS
whatever the precision or chain count, and at M=8 a 17408x5120 matmul is 1.4
GFLOP = 0.265 ms of MMA alone — the same as its memory floor. The kernel is ALU-
issue-bound (the weight stream through this access pattern alone runs at 177
GB/s), so every non-MMA instruction in the loop is the cost; what remains is ~70
ALU ops per tile-group against 8 MMAs. Threadgroup/tile shapes, software
prefetch, explicit bfe/bfi, tile-interleaved chains were all measured flat or
worse.

Measured (M4 Pro, mlx 0.32, dependent chains, rotated weights; ms per call,
stock M=1 / stock at M / this kernel; the staged kernel in brackets):
  17408x5120 3-bit  M=8: 0.207 / 0.796 / 0.380 [0.485]   2.09x stock, 1.28x staged
  5120x34816 3-bit  M=8: 0.371 / 1.503 / 0.688 [0.917]   2.18x / 1.33x
  5120x248320 5-bit M=8: 3.43  / 11.1  / 4.82  [5.91]    2.30x / 1.23x
  M=5 already wins 1.2-1.4x on every shape (the staged kernel tied stock there);
  the kernel is flat in M, so widths 4..8 cost the same.

Numerics: fp32 accumulation in a different order than qmm — 1-2 bf16 ulps apart,
the same acceptance class as every other kernel swap here (fused attention,
fastpath compile): per-token greedy-correct under the verify loop, NOT bit-identical
to the stock kernel. The probe (:func:`calibrate`) checks both numerics and speed
per (shape, M) on THIS chip x mlx version and keeps only widths where the kernel
actually wins — the result is cached under ~/.cache/chad/qmm_mma/. CHAD_NO_QMM_MMA
disables; CHAD_QMM_MMA_RECAL forces a fresh probe.

Measurement notes (they cost real time upstream, don't relearn): time DEPENDENT
chains only — independent launches overlap and hide the critical path; rotate
across same-shape weights so nothing stays cache-resident (the probe rotates across
the model's own layers, allocating nothing).
"""

import json
import os
import platform
import time
from typing import Any, Optional

from . import config
from .diag import log

M_MAX = 8            # one MMA tile
N_MIN = 4096         # fewer output columns = too few threadgroups to fill the GPU
_CACHE_DIR = os.path.expanduser("~/.cache/chad/qmm_mma")
_SUPPORTED_BITS = (3, 4, 5, 6, 8)

_SRC = r"""
    const int K = KD, N = ND, M = MD;
    const int KPS = KD / SG;                // K-span per simdgroup (split-K)
    const int GW = 64 * BITS / 32;          // packed words per 64-value group
    const int WPH = (BITS == 4) ? 1 : 2;    // words per 8-value half (after alignment)
    // bf16(2^BITS + v) == MAGIC | (v << (7 - BITS)), exact for BITS <= 7;
    // 8-bit takes the exact bf16 of v itself (POW = 0).
    const uint MAGIC = (BITS == 8) ? 0u : ((uint)(127 + BITS) << 7);
    const uint MAGIC2 = MAGIC | (MAGIC << 16);
    const float POW = (BITS == 8) ? 0.0f : (float)(1 << BITS);

    uint tid  = thread_position_in_threadgroup.x;
    uint tgid = threadgroup_position_in_grid.x;
    uint sg   = tid >> 5;
    uint lane = tid & 31;
    // simdgroup_matrix<T,8,8> fragment layout: lane -> row fm, columns fn, fn+1
    uint qid = lane >> 2;
    int fm = (int)((qid & 4) + ((lane >> 1) & 3));        // lane bits 1,2,4
    int fn = (int)(((qid & 2) << 1) + ((lane & 1) << 1));  // lane bits 0,3

    int n0 = (int)tgid * 8 * TILES;
    threadgroup float red[SG * TILES * 64];

    float acc0[TILES], acc1[TILES];
#pragma clang loop unroll(full)
    for (int t = 0; t < TILES; ++t) { acc0[t] = 0.0f; acc1[t] = 0.0f; }

    // x rows this lane feeds into the B fragment (m = fn, fn+1); rows >= M are zero
    const bool xa_ok = fn < M, xb_ok = fn + 1 < M;
    const device bfloat16_t* xa_p = x + (size_t)(xa_ok ? fn : 0) * K + fm * 8;
    const device bfloat16_t* xb_p = x + (size_t)(xb_ok ? fn + 1 : 0) * K + fm * 8;
    // this lane's 16 contiguous weight values start at value fn*8 of each group
    // (k inside a group is permuted as k = c*8 + kt, identically for W and x)
    const int hoff0 = (fn * 8) * BITS, hoff1 = (fn * 8 + 8) * BITS;
    const int hw0 = hoff0 >> 5, hs0 = hoff0 & 31;
    const int hw1 = hoff1 >> 5, hs1 = hoff1 & 31;

    const int gbeg = (int)sg * (KPS / 64);
    const int gend = gbeg + KPS / 64;
    for (int g = gbeg; g < gend; ++g) {
        uint4 xa = xa_ok ? *((const device uint4*)(xa_p + g * 64)) : uint4(0u);
        uint4 xb = xb_ok ? *((const device uint4*)(xb_p + g * 64)) : uint4(0u);
        simdgroup_matrix<bfloat16_t, 8, 8> B[8];
        float sa = 0.0f, sb = 0.0f;
#pragma clang loop unroll(full)
        for (int kt = 0; kt < 8; ++kt) {
            uint wa = (kt & 1) ? (xa[kt >> 1] & 0xFFFF0000u) : (xa[kt >> 1] << 16);
            uint wb = (kt & 1) ? (xb[kt >> 1] & 0xFFFF0000u) : (xb[kt >> 1] << 16);
            sa += as_type<float>(wa);
            sb += as_type<float>(wb);
            thread auto& eb = B[kt].thread_elements();
            reinterpret_cast<thread uint&>(eb) = (wa >> 16) | wb;
        }
        // group sums of x rows fn / fn+1: reduce over the 8 lanes that differ in fm
        sa += simd_shuffle_xor(sa, 2u);  sb += simd_shuffle_xor(sb, 2u);
        sa += simd_shuffle_xor(sa, 4u);  sb += simd_shuffle_xor(sb, 4u);
        sa += simd_shuffle_xor(sa, 16u); sb += simd_shuffle_xor(sb, 16u);

#pragma clang loop unroll(full)
        for (int t = 0; t < TILES; ++t) {
            int n = n0 + t * 8 + fm;
            bool ok = n < N;
            int nn = ok ? n : 0;
            float s  = (float)sc[(size_t)nn * (K / 64) + g];
            float bb = (float)bi[(size_t)nn * (K / 64) + g];
            const device uint* wr = w + (size_t)nn * (K * BITS / 32) + (size_t)g * GW;
            ulong win0, win1;
            if (WPH == 1) {
                win0 = (ulong)wr[hw0];
                win1 = (ulong)wr[hw1];
            } else {
                win0 = ((ulong)wr[hw0] | ((ulong)wr[hw0 + 1] << 32)) >> hs0;
                win1 = ((ulong)wr[hw1] | ((ulong)wr[hw1 + 1] << 32)) >> hs1;
            }
            simdgroup_matrix<float, 8, 8> Cg = simdgroup_matrix<float, 8, 8>(0);
            simdgroup_matrix<bfloat16_t, 8, 8> A;
            thread auto& ea = A.thread_elements();
#pragma clang loop unroll(full)
            for (int kt = 0; kt < 8; ++kt) {
                uint v0 = (uint)(win0 >> (kt * BITS)) & ((1u << BITS) - 1u);
                uint v1 = (uint)(win1 >> (kt * BITS)) & ((1u << BITS) - 1u);
                if (BITS == 8) {
                    ea[0] = (bfloat16_t)((float)v0);
                    ea[1] = (bfloat16_t)((float)v1);
                } else {
                    reinterpret_cast<thread uint&>(ea) =
                        MAGIC2 | (v0 << (7 - BITS)) | (v1 << (23 - BITS));
                }
                simdgroup_multiply_accumulate(Cg, A, B[kt], Cg);
            }
            thread auto& cg = Cg.thread_elements();
            float bbs = ok ? (bb - POW * s) : 0.0f;
            s = ok ? s : 0.0f;
            acc0[t] += s * cg[0] + bbs * sa;
            acc1[t] += s * cg[1] + bbs * sb;
        }
    }

#pragma clang loop unroll(full)
    for (int t = 0; t < TILES; ++t) {
        red[(sg * TILES + t) * 64 + fm * 8 + fn]     = acc0[t];
        red[(sg * TILES + t) * 64 + fm * 8 + fn + 1] = acc1[t];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = (int)tid; i < TILES * 64; i += 32 * SG) {
        int t = i >> 6, nl = (i >> 3) & 7, m = i & 7;
        int n = n0 + t * 8 + nl;
        if (m < M && n < N) {
            float v = 0.0f;
            for (int q = 0; q < SG; ++q) v += red[(q * TILES + t) * 64 + nl * 8 + m];
            out[(size_t)m * N + n] = (bfloat16_t)v;
        }
    }
"""

_SG = 8                  # simdgroups per threadgroup (split-K factor)
_TILES = 4               # 8-column tiles per simdgroup
_TG = 32 * _SG
_KERNEL_VERSION = 4      # bump when the kernel changes: the probe cache is keyed on it
_kernels: dict = {}


def _kernel(bits: int):
    k = _kernels.get(bits)
    if k is None:
        import mlx.core as mx
        k = mx.fast.metal_kernel(
            name=f"chad_qmm_mma{bits}",
            input_names=["x", "w", "sc", "bi"],
            output_names=["out"],
            source=_SRC.replace("BITS", str(bits)),
        )
        _kernels[bits] = k
    return k


def mma(x, wq, sc, bi, M: int, N: int, K: int, bits: int):
    """Raw kernel call: x is (M, K) bf16, M <= 8 (rows past M are zero in-kernel),
    returns (M, N) bf16."""
    cols = 8 * _TILES
    (out,) = _kernel(bits)(
        inputs=[x, wq, sc, bi],
        template=[("KD", K), ("ND", N), ("MD", M), ("TILES", _TILES), ("SG", _SG)],
        output_shapes=[(M, N)], output_dtypes=[x.dtype],
        grid=(((N + cols - 1) // cols) * _TG, 1, 1), threadgroup=(_TG, 1, 1),
    )
    return out


def shape_ok(K: int, N: int, bits: int, group_size: int) -> bool:
    return (bits in _SUPPORTED_BITS and group_size == 64 and N >= N_MIN
            and K % 512 == 0)


# Verified dispatch table: (K, N, bits) -> smallest M the kernel wins at (M_MAX
# bounds the top). Empty until calibrate() runs; a missing shape runs stock.
_WINS: dict = {}


def qmm(x, wq, sc, bi, group_size: int, bits: int):
    """Drop-in for ``mx.quantized_matmul(x, wq, scales=sc, biases=bi, transpose=True,
    group_size=group_size, bits=bits)`` that takes the MMA kernel on verified
    (shape, width) pairs and the stock kernel everywhere else."""
    import mlx.core as mx
    if _WINS and x.dtype == mx.bfloat16:
        K = x.shape[-1]
        N = wq.shape[0]
        m_min = _WINS.get((K, N, bits))
        if m_min is not None:
            M = 1
            for d in x.shape[:-1]:
                M *= d
            if m_min <= M <= M_MAX:
                return mma(x.reshape(M, K), wq, sc, bi, M, N, K, bits
                           ).reshape(*x.shape[:-1], N)
    return mx.quantized_matmul(x, wq, scales=sc, biases=bi, transpose=True,
                               group_size=group_size, bits=bits)


def _install_patch() -> None:
    """Route every ``nn.QuantizedLinear`` through :func:`qmm` (class patch, once).
    Shapes outside the verified table fall straight through to the stock call."""
    import mlx.nn as nn
    if getattr(nn.QuantizedLinear.__call__, "_chad_qmm_mma", False):
        return
    stock = nn.QuantizedLinear.__call__

    def call(self, x):
        if not _WINS or getattr(self, "mode", "affine") != "affine" \
                or "biases" not in self or self["weight"].ndim != 2:
            return stock(self, x)
        y = qmm(x, self["weight"], self["scales"], self["biases"],
                self.group_size, self.bits)
        if "bias" in self:
            y = y + self["bias"]
        return y

    call._chad_qmm_mma = True  # type: ignore[attr-defined]
    nn.QuantizedLinear.__call__ = call  # type: ignore[method-assign]


# ------------------------------------------------------------------ calibration

_CHAIN = 12          # dependent steps per timed eval
_EVALS = 4           # first is warmup, median of the rest
_MIN_GAIN = 1.10     # the kernel must beat qmm by this at a width to own it
_REL_TOL = 0.02      # max|kernel - qmm| <= _REL_TOL * max|qmm| (measured ~0.007)


def _time_chain(step, x0) -> float:
    import mlx.core as mx
    times = []
    for _ in range(_EVALS):
        x = x0
        t0 = time.perf_counter()
        for t in range(_CHAIN):
            y = step(x, t)
            x = x0 + mx.mean(y).astype(x0.dtype) * 1e-20  # real dependency, ~no drift
        mx.eval(y, x)
        times.append((time.perf_counter() - t0) / _CHAIN)
    rest = sorted(times[1:])
    return rest[len(rest) // 2]


def _eligible_groups(*models) -> dict:
    """(K, N, bits) -> [(wq, sc, bi), ...] for every eligible QuantizedLinear AND
    every fused weight the fastpath stacked (they are plain arrays, not modules)."""
    import mlx.nn as nn
    groups: dict = {}

    def add(wq, sc, bi, gs, bits):
        if wq.ndim != 2:
            return          # fastpath placeholder left behind a concatenated original
        bits, gs = int(bits), int(gs)
        N = int(wq.shape[0])
        K = int(wq.shape[1]) * 32 // bits
        if shape_ok(K, N, bits, gs):
            groups.setdefault((K, N, bits), []).append((wq, sc, bi))

    for model in models:
        if model is None:
            continue
        for _, mod in model.named_modules():
            if (isinstance(mod, nn.QuantizedLinear)
                    and getattr(mod, "mode", "affine") == "affine"
                    and "biases" in mod):
                add(mod["weight"], mod["scales"], mod["biases"], mod.group_size, mod.bits)
            if hasattr(mod, "_fused_w"):
                add(mod._fused_w, mod._fused_s, mod._fused_b, mod._fused_gs,
                    mod._fused_bits)
    return groups


def _cache_key() -> str:
    import mlx.core as mx
    try:
        chip = mx.device_info().get("device_name", "metal")
    except Exception:  # noqa: BLE001
        chip = platform.machine()
    return f"{chip}-mlx{mx.__version__}-k{_KERNEL_VERSION}".replace(" ", "_").replace("/", "_")


def measure(groups: dict, verbose: bool = False) -> dict:
    """For every shape: numerics at each width, then a dependent-chain race at each
    width M=2..8 (rotating across the shape's own weights). Returns
    {(K, N, bits): m_min} for shapes where the kernel wins from m_min up through
    M_MAX (a shape that wins only in the middle is kept from its first win; a
    shape that never wins is absent)."""
    import mlx.core as mx
    wins: dict = {}
    for (K, N, bits), ws in groups.items():
        wq, sc, bi = ws[0]
        ok = True
        for M in range(2, M_MAX + 1):
            x = (mx.random.normal((M, K)) * 0.1).astype(mx.bfloat16)
            ref = mx.quantized_matmul(x, wq, scales=sc, biases=bi, transpose=True,
                                      group_size=64, bits=bits).astype(mx.float32)
            got = mma(x, wq, sc, bi, M, N, K, bits).astype(mx.float32)
            diff = mx.max(mx.abs(ref - got))
            scale = mx.max(mx.abs(ref))
            mx.eval(diff, scale)
            if diff.item() > _REL_TOL * max(scale.item(), 1.0):
                ok = False
                break
        if not ok:
            log.warning("qmm_mma: %dx%d b%d rejected (numerics)", K, N, bits)
            continue
        ratios = {}
        for M in range(2, M_MAX + 1):
            x = (mx.random.normal((M, K)) * 0.1).astype(mx.bfloat16)
            mx.eval(x)

            def q_step(xx, t, _ws=ws, _b=bits):
                w = _ws[t % len(_ws)]
                return mx.quantized_matmul(xx, w[0], scales=w[1], biases=w[2],
                                           transpose=True, group_size=64, bits=_b)

            def k_step(xx, t, _ws=ws, _M=M, _N=N, _K=K, _b=bits):
                w = _ws[t % len(_ws)]
                return mma(xx, w[0], w[1], w[2], _M, _N, _K, _b)

            tq = min(_time_chain(q_step, x), _time_chain(q_step, x))
            tk = min(_time_chain(k_step, x), _time_chain(k_step, x))
            ratios[M] = tq / tk
        first = next((M for M in range(2, M_MAX + 1)
                      if all(ratios[m] >= _MIN_GAIN for m in range(M, M_MAX + 1))),
                     None)
        msg = " ".join(f"M{M}={r:.2f}x" for M, r in ratios.items())
        if first is not None:
            wins[(K, N, bits)] = first
        log.info("qmm_mma: %dx%d b%d %s -> %s", K, N, bits, msg,
                 f"on from M={first}" if first else "off")
        if verbose:
            print(f"  qmm_mma {K}x{N} b{bits}: {msg} -> "
                  f"{'on from M=%d' % first if first else 'off'}", flush=True)
    return wins


def calibrate(*models, verbose: bool = False) -> dict:
    """Probe (or load the cached probe for) every eligible shape in `models`,
    install the dispatch, and return the win table. Cached per chip x mlx
    version x shape set; CHAD_QMM_MMA_RECAL=1 re-measures."""
    groups = _eligible_groups(*models)
    if not groups:
        return {}
    keys = sorted(f"{K}x{N}b{b}" for (K, N, b) in groups)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, _cache_key() + ".json")
    cached: dict = {}
    if os.path.isfile(path) and not config.flag("CHAD_QMM_MMA_RECAL"):
        try:
            with open(path) as f:
                cached = json.load(f)
        except Exception:  # noqa: BLE001
            cached = {}
    missing = {k: v for k, v in groups.items() if f"{k[0]}x{k[1]}b{k[2]}" not in cached}
    if missing:
        t0 = time.time()
        fresh = measure(missing, verbose=verbose)
        for k in missing:
            cached[f"{k[0]}x{k[1]}b{k[2]}"] = fresh.get(k, 0)
        try:
            with open(path, "w") as f:
                json.dump(cached, f, indent=1, sort_keys=True)
        except Exception:  # noqa: BLE001
            pass
        log.info("qmm_mma: probed %d shape(s) in %.1fs", len(missing), time.time() - t0)
    wins = {k: int(cached[f"{k[0]}x{k[1]}b{k[2]}"]) for k in groups
            if int(cached.get(f"{k[0]}x{k[1]}b{k[2]}", 0) or 0) > 0}
    _WINS.clear()
    _WINS.update(wins)
    if wins:
        _install_patch()
        log.info("qmm_mma: small-M MMA verify kernel on for %d/%d shapes (%s)",
                 len(wins), len(keys),
                 ", ".join(f"{K}x{N}b{b}@M>={m}" for (K, N, b), m in sorted(wins.items())))
    return wins


def install(*models, verbose: bool = False) -> bool:
    """Engine entry point: probe + patch unless disabled. Never raises."""
    if config.flag("CHAD_NO_QMM_MMA"):
        return False
    try:
        return bool(calibrate(*models, verbose=verbose))
    except Exception as e:  # noqa: BLE001 — a speed kernel never breaks a load
        log.warning("qmm_mma install failed (%s); stock quantized_matmul", e)
        _WINS.clear()
        return False


def active() -> bool:
    return bool(_WINS)


def disable() -> None:
    """Drop the win table: every call falls through to the stock kernel (the
    class patch stays, inert). For A/B arms and tests."""
    _WINS.clear()


def set_wins(wins: Optional[dict]) -> None:
    """Force a win table (tests / A/B arms): {(K, N, bits): m_min}."""
    _WINS.clear()
    if wins:
        _WINS.update(wins)
        _install_patch()


def stock(x, wq, sc, bi, group_size: int, bits: int) -> Any:
    import mlx.core as mx
    return mx.quantized_matmul(x, wq, scales=sc, biases=bi, transpose=True,
                               group_size=group_size, bits=bits)
