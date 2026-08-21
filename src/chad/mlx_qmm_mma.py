"""Small-M quantized matmul for speculative verify: an MMA kernel that reads each
weight group ONCE for all rows.

``mx.quantized_matmul`` is at roofline for M=1 (decode) and tiles well from M≈13
up (prefill), but in between — exactly where speculative verification lives, M =
draft width + 1 — its cost grows nearly linearly in M: the GEMV path re-pays the
weight read per row. Measured on this engine's verify ladder: ~33 ms per extra
row on the 3-bit 27B (a 56 ms serial step), which is why the DFlash block's width
was pinned at 4 when its acceptance would carry 7.

The kernel here is avlp12's ``qmm_mma4`` (github.com/avlp12/mlx-lm ``fast_qmm.py``,
MIT) as vendored and re-dispatched in ARahim3/mlx-dspark (``small_m_qmm.py``, MIT,
Copyright (c) 2026 ARahim3): an 8x8 ``simdgroup_matrix`` MMA tile covers M <= 8
exactly, so each quantized weight group is dequantized once and reused by every
row; K is split across the 8 simdgroups of a threadgroup. Changed here: the
dequant unpack is generic over the affine bit width — MLX packs every width as a
continuous little-endian bitstream per row (verified for 3/4/5/6/8), so one unpack
serves the shipped 3-bit body and 5-bit ``lm_head`` as well as 4/8-bit targets —
(the unpack pulls each 8-value half through one 64-bit window so every shift is a
compile-time constant after unrolling, and each barrier pair covers two quant
groups so both groups' loads are in flight together), and the dispatch is a persistent patch
gated by a per-shape, per-width runtime probe instead of a context manager.
Measured (M4 Pro, mlx 0.32, dependent chains, rotated weights): 17408x5120 3-bit
M=6 1.31x / M=8 1.64x; 5120x248320 5-bit (lm_head) M=6 1.55x / M=8 1.87x; 4-bit
M=6 1.41x / M=8 1.74x (upstream's 64-k form: 1.30x / 1.60x). Below M=5 the stock GEMV path wins.

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
    const int KPS = KD / 8;                 // K-span per simdgroup (split-K)
    const int GW = 64 * BITS / 32;          // packed words per 64-value group
    const int NG = CHUNK / 64;              // quant groups per barrier pair (1 or 2)

    uint tid  = thread_position_in_threadgroup.x;
    uint tgid = threadgroup_position_in_grid.x;
    uint sg   = tid >> 5;
    uint lane = tid & 31;

    int n0 = (int)tgid * 8;                 // one threadgroup -> 8 output columns

    threadgroup bfloat16_t bs[8 * 1024];    // per-simdgroup CHUNK k x 8 n dequant stage
    threadgroup float red[8 * 64];          // cross-simdgroup reduction

    simdgroup_matrix<float, 8, 8> C = simdgroup_matrix<float, 8, 8>(0);
    threadgroup bfloat16_t* bt = bs + sg * 1024;

    int kbeg = (int)sg * KPS;
    int j  = (int)(lane & 7);
    int kq = (int)(lane >> 3);              // this lane's 16-value quarter of a group
    int n  = n0 + j;
    const int base = kq * 16 * BITS;
    const int w0 = base >> 5;
    // CHUNK k per barrier pair: with CHUNK=128 both groups' packed words are in
    // flight before any dequant math (the 64-k form exposed one load latency per
    // iteration; measured +3-8% on the real shapes).
    for (int kk = 0; kk < KPS; kk += CHUNK) {
        int ka = kbeg + kk;
        if (n < N) {
#pragma clang loop unroll(full)
            for (int c = 0; c < NG; ++c) {
                int g = (ka >> 6) + c;
                float s  = (float)sc[(size_t)n * (K / 64) + g];
                float bb = (float)bi[(size_t)n * (K / 64) + g];
                // Generic affine unpack: value t of the group sits at bit t*BITS of
                // the row's little-endian word stream; this lane's 16 values start
                // at bit kq*16*BITS and span at most 4 words for BITS <= 8.
                const device uint* wr = w + (size_t)n * (K * BITS / 32) + (size_t)g * GW;
                uint p[4];
#pragma clang loop unroll(full)
                for (int u = 0; u < 4; ++u) p[u] = (w0 + u < GW) ? wr[w0 + u] : 0u;
                // Two 8-value halves, each pulled from one 64-bit window: for every
                // width <= 8, 8*BITS plus the half's in-word shift fits in 64 bits,
                // so after unrolling every shift and mask is a compile-time constant
                // (a per-value runtime word index ran ~40% slower).
#pragma clang loop unroll(full)
                for (int h = 0; h < 2; ++h) {
                    int off = base + h * 8 * BITS - (w0 << 5);
                    int wi = off >> 5, sh = off & 31;
                    ulong win = ((ulong)p[wi] | ((ulong)p[wi + 1] << 32)) >> sh;
#pragma clang loop unroll(full)
                    for (int t = 0; t < 8; ++t) {
                        uint v = (uint)(win >> (t * BITS)) & ((1u << BITS) - 1u);
                        bt[(c * 64 + kq * 16 + h * 8 + t) * 8 + j] =
                            (bfloat16_t)((float)v * s + bb);
                    }
                }
            }
        } else {
            for (int t = 0; t < 16 * NG; ++t)
                bt[((t >> 4) * 64 + kq * 16 + (t & 15)) * 8 + j] = (bfloat16_t)0;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<bfloat16_t, 8, 8> A, B;
#pragma clang loop unroll(full)
        for (int kt = 0; kt < CHUNK / 8; ++kt) {
            simdgroup_load(A, x + ka + kt * 8, K);   // x rows 0..7 (padded to 8)
            simdgroup_load(B, bt + kt * 64, 8);
            simdgroup_multiply_accumulate(C, A, B, C);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(C, red + sg * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = (int)tid; i < 64; i += 256) {
        int m = i >> 3, jj = i & 7;
        int nn = n0 + jj;
        if (m < M && nn < N) {
            float v = 0.0f;
            for (int q = 0; q < 8; ++q) v += red[q * 64 + i];
            out[(size_t)m * N + nn] = (bfloat16_t)v;
        }
    }
"""

_TG = 256
_KERNEL_VERSION = 3      # bump when the kernel changes: the probe cache is keyed on it
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


def mma(x8, wq, sc, bi, M: int, N: int, K: int, bits: int):
    """Raw kernel call: x8 is (8, K) bf16 (rows >= M ignored), returns (M, N) bf16."""
    (out,) = _kernel(bits)(
        inputs=[x8, wq, sc, bi],
        template=[("KD", K), ("ND", N), ("MD", M), ("CHUNK", 128 if K % 1024 == 0 else 64)],
        output_shapes=[(M, N)], output_dtypes=[x8.dtype],
        grid=(((N + 7) // 8) * _TG, 1, 1), threadgroup=(_TG, 1, 1),
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
                flat = x.reshape(M, K)
                if M < 8:
                    flat = mx.concatenate(
                        [flat, mx.zeros((8 - M, K), dtype=flat.dtype)], axis=0)
                return mma(flat, wq, sc, bi, M, N, K, bits).reshape(*x.shape[:-1], N)
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
        chip = mx.metal.device_info().get("device_name", "metal")
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
            x8 = x if M == 8 else mx.concatenate(
                [x, mx.zeros((8 - M, K), dtype=x.dtype)], axis=0)
            ref = mx.quantized_matmul(x, wq, scales=sc, biases=bi, transpose=True,
                                      group_size=64, bits=bits).astype(mx.float32)
            got = mma(x8, wq, sc, bi, M, N, K, bits).astype(mx.float32)
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
            x8 = x if M == 8 else mx.concatenate(
                [x, mx.zeros((8 - M, K), dtype=x.dtype)], axis=0)
            mx.eval(x, x8)

            def q_step(xx, t, _ws=ws, _b=bits):
                w = _ws[t % len(_ws)]
                return mx.quantized_matmul(xx, w[0], scales=w[1], biases=w[2],
                                           transpose=True, group_size=64, bits=_b)

            def k_step(xx, t, _ws=ws, _M=M, _N=N, _K=K, _b=bits):
                w = _ws[t % len(_ws)]
                return mma(xx, w[0], w[1], w[2], _M, _N, _K, _b)

            tq = min(_time_chain(q_step, x), _time_chain(q_step, x))
            tk = min(_time_chain(k_step, x8), _time_chain(k_step, x8))
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
