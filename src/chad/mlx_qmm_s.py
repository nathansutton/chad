"""Fused small-batch quantized matmul: qmv with S right-hand sides.

Why this exists: speculative decoding (MTP / draft-model) verifies k drafted
tokens in one S = k+1 forward, betting that reading the weights once for S
tokens costs about what it costs for one. On mlx 0.32 that bet loses — there
is no kernel between qmv (S=1) and tiled qmm (S large), so S in [2, 8] pays
almost per-row: measured on the 27B MLP shapes (M4 Pro, 3-bit gs64),

    S=1 0.30 ms   S=2 0.42   S=4 0.51   S=6 0.80   S=8 0.95    (mx 0.32.0)

i.e. S=6 is 2.7x S=1 on a bandwidth-bound op whose weight traffic is
identical. That scaling is the entire reason MTP verify forwards cost ~2-3x
a plain decode step and speculation washes out (measured: k=5 reaches 2.6
tok/forward yet DROPS throughput 17.7 -> 11.8 tok/s).

This kernel is the qmv structure — each simdgroup owns 4 output rows, lanes
split the K axis, one simd_sum per (row, s) at the end — with the one change
that matters: the dequantized weight chunk is applied to S input rows while
it sits in registers. Weight bytes are read exactly once regardless of S; the
extra cost is S-1 register-resident FMA passes, which is the part the M4 has
headroom for until S ~ 8 (the ALU/bandwidth crossover).

Scope (fall back to mx.quantized_matmul outside it — correctness first):
  - affine mode, group_size 64, bits in {2, 3, 4, 6, 8}
  - transpose=True (the nn.QuantizedLinear orientation), fp16 activations
  - batch 1, 2 <= S <= 8, K % 256 == 0, N % 8 == 0
  (every projection in the Qwen3.8-27B text stack qualifies: K in
   {5120, 6144, 10240, 17408}, N from 48 to 248320.)

install() routes eligible calls here from BOTH call surfaces:
  - nn.QuantizedLinear.__call__ (attention, down_proj, lm_head, …)
  - mlx_fastpath's fused-concat sites (GDN in_proj, MLP gate|up), via the
    fastpath QMM hook.
S=1 decode and large-S prefill are untouched.
"""

import os
from typing import Any, Optional

from . import config
from .diag import log

_GS = 64                       # the only group size the kernel is built for
_BITS = (2, 3, 4, 6, 8)        # byte-aligned 8-value lanes exist for these
_MAX_S = 8

_HEADER = """
// Unpack 8 consecutive affine-quantized values starting at a byte-aligned
// pointer. 8 values * BITS bits is a whole number of bytes for BITS in
// {2,3,4,6,8}, and a lane's 8 values never straddle a 64-value scale group,
// so the caller applies one scale/bias per chunk. Layouts mirror
// mlx/backend/metal/kernels/quantized.h (LSB-first within each byte).
template <int BITS>
inline void chad_unpack8(const device uchar* w, thread float* q) {
  if (BITS == 2) {
    for (int i = 0; i < 2; ++i) {
      uchar b = w[i];
      q[4*i+0] = float(b & 0x3);
      q[4*i+1] = float((b >> 2) & 0x3);
      q[4*i+2] = float((b >> 4) & 0x3);
      q[4*i+3] = float((b >> 6) & 0x3);
    }
  } else if (BITS == 3) {
    uchar b0 = w[0], b1 = w[1], b2 = w[2];
    q[0] = float(b0 & 0x7);
    q[1] = float((b0 >> 3) & 0x7);
    q[2] = float(((b0 >> 6) | (b1 << 2)) & 0x7);
    q[3] = float((b1 >> 1) & 0x7);
    q[4] = float((b1 >> 4) & 0x7);
    q[5] = float(((b1 >> 7) | (b2 << 1)) & 0x7);
    q[6] = float((b2 >> 2) & 0x7);
    q[7] = float((b2 >> 5) & 0x7);
  } else if (BITS == 4) {
    for (int i = 0; i < 4; ++i) {
      uchar b = w[i];
      q[2*i]   = float(b & 0xf);
      q[2*i+1] = float(b >> 4);
    }
  } else if (BITS == 6) {
    for (int i = 0; i < 2; ++i) {
      uchar b0 = w[3*i], b1 = w[3*i+1], b2 = w[3*i+2];
      q[4*i+0] = float(b0 & 0x3f);
      q[4*i+1] = float(((b0 >> 6) | (b1 << 2)) & 0x3f);
      q[4*i+2] = float(((b1 >> 4) | (b2 << 4)) & 0x3f);
      q[4*i+3] = float((b2 >> 2) & 0x3f);
    }
  } else {  // BITS == 8
    for (int i = 0; i < 8; ++i) q[i] = float(w[i]);
  }
}
"""

# Kernel body. Grid: one threadgroup of 64 threads (2 simdgroups) per 8 output
# rows; simdgroup r-block = 4 rows. Lanes split K into 256-value iterations
# (32 lanes x 8 values); a lane's 8-value chunk lives in one scale group.
# y is (S, N) fp16; accumulation fp32.
_SRC = """
    uint tg = threadgroup_position_in_grid.x;
    uint simd = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    const uint N = params[0];
    const uint K = params[1];
    uint row0 = tg * 8 + simd * 4;
    if (row0 >= N) return;
    const device uchar* wb = (const device uchar*)w;
    const uint row_bytes = (K * BITS) / 8;
    const uint gstride = K / 64;

    float acc[4][S];
    for (int r = 0; r < 4; ++r)
      for (int s = 0; s < S; ++s) acc[r][s] = 0.0f;

    const uint iters = K / 256;
    for (uint it = 0; it < iters; ++it) {
      uint v0 = it * 256 + lane * 8;
      uint g = v0 / 64;

      float qv[4][8];
      float sc[4], bi[4];
      for (int r = 0; r < 4; ++r) {
        uint row = row0 + r;
        chad_unpack8<BITS>(wb + row * row_bytes + (v0 * BITS) / 8, qv[r]);
        sc[r] = float(scales[row * gstride + g]);
        bi[r] = float(biases[row * gstride + g]);
      }
      for (int s = 0; s < S; ++s) {
        const device half* xp = x + s * K + v0;
        float xv[8];
        float xs = 0.0f;
        for (int i = 0; i < 8; ++i) { xv[i] = float(xp[i]); xs += xv[i]; }
        for (int r = 0; r < 4; ++r) {
          float d = 0.0f;
          for (int i = 0; i < 8; ++i) d += qv[r][i] * xv[i];
          acc[r][s] += sc[r] * d + bi[r] * xs;
        }
      }
    }

    for (int r = 0; r < 4; ++r) {
      for (int s = 0; s < S; ++s) {
        float v = simd_sum(acc[r][s]);
        if (lane == 0) y[s * N + row0 + r] = half(v);
      }
    }
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        import mlx.core as mx
        _kernel = mx.fast.metal_kernel(
            name="chad_qmm_s",
            input_names=["x", "w", "scales", "biases", "params"],
            output_names=["y"],
            header=_HEADER,
            source=_SRC,
        )
    return _kernel


def eligible(x: Any, biases: Any, group_size: int, bits: int,
             mode: str = "affine") -> bool:
    import mlx.core as mx
    if mode != "affine" or group_size != _GS or bits not in _BITS:
        return False
    if biases is None or x.dtype != mx.float16 or x.ndim != 3:
        return False
    B, S, K = x.shape
    return B == 1 and 2 <= S <= _MAX_S and K % 256 == 0


def qmm_s(x: Any, w: Any, scales: Any, biases: Any, bits: int) -> Any:
    """The raw kernel call: x (1, S, K) fp16 -> (1, S, N) fp16. Caller has
    already checked eligible(); N % 8 is checked here (falls back)."""
    import mlx.core as mx
    _, S, K = x.shape
    N = w.shape[0]
    if N % 8:
        return mx.quantized_matmul(
            x, w, scales=scales, biases=biases, transpose=True,
            group_size=_GS, bits=bits)
    params = mx.array([N, K], dtype=mx.uint32)
    (y,) = _get_kernel()(
        inputs=[x.reshape(S, K), w, scales, biases, params],
        template=[("BITS", bits), ("S", S)],
        grid=((N // 8) * 64, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(S, N)],
        output_dtypes=[x.dtype],
    )
    return y.reshape(1, S, N)


def qmm(x: Any, w: Any, *, scales: Any, biases: Any = None,
        transpose: bool = True, group_size: int = 64, bits: int = 4,
        mode: str = "affine") -> Any:
    """Drop-in for mx.quantized_matmul that routes the small-batch decode
    shape to the fused kernel and everything else to the stock op."""
    import mlx.core as mx
    if transpose and eligible(x, biases, group_size, bits, mode):
        return qmm_s(x, w, scales, biases, bits)
    return mx.quantized_matmul(
        x, w, scales=scales, biases=biases, transpose=transpose,
        group_size=group_size, bits=bits, mode=mode)


def install() -> bool:
    """Route both quantized-matmul call surfaces through qmm(). Class-level
    and idempotent, mirroring the other fastpath patches: ineligible shapes
    take the stock op inside qmm(), so foreign models are unaffected."""
    if config.flag("CHAD_NO_QMMS"):
        return False
    try:
        import mlx.core as mx  # noqa: F401
        import mlx.nn as nn

        if not getattr(nn.QuantizedLinear.__call__, "_chad_qmm_s", False):
            def call(self, xx):
                out = qmm(
                    xx, self["weight"], scales=self["scales"],
                    biases=self.get("biases"), transpose=True,
                    group_size=self.group_size, bits=self.bits,
                    mode=self.mode)
                if "bias" in self:
                    out = out + self["bias"]
                return out
            call._chad_qmm_s = True  # type: ignore[attr-defined]
            nn.QuantizedLinear.__call__ = call  # type: ignore[method-assign]

        from . import mlx_fastpath
        mlx_fastpath.QMM = qmm
        log.info("qmm_s installed: fused small-batch quantized matmul "
                 "(verify forwards S 2..%d)", _MAX_S)
        return True
    except Exception as e:  # noqa: BLE001 — perf path must never break loading
        log.warning("qmm_s install failed (%s); running stock", e)
        return False
