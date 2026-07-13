"""Fused quantized-KV decode attention (plan 081 roadmap item 1).

mlx_lm's quantized-KV attention (`kv_bits=8`) is manually unfused — two
`quantized_matmul`s around a materialized softmax — and measured 2.3-2.6x
SLOWER than fused fp16 attention despite reading half the bytes (plan 081:
53->38 tok/s @32k). That made cache quantization a pure loss, so chad ships
with it off and the governor's ctx_limit pays full fp16 KV rates (~20.5
KB/token on the 35B).

This module JIT-compiles (mx.fast.metal_kernel — no wheel rebuild) a fused
online-softmax scan over the 8-bit group-64 affine QuantizedKVCache layout,
structured like mlx's sdpa_vector_2pass with one addition: each threadgroup
cooperatively dequantizes a CHUNK of 8 positions into threadgroup memory
ONCE, double-buffered so the next chunk's packed loads overlap the current
chunk's math, and the 8 GQA q-head simdgroups consume cheap staged rows.
(A naive port that dequantized per-head was ALU-bound and lost to fp16;
sharing the dequant is what flips it. See private/plans/081 for the
variant-by-variant record.)

Measured on the target shape (q 16h x 256d, kv 2h, GQA-8, bf16, M4 Pro):
vs stock fused fp16 sdpa_vector: 1.17x @8k, 1.31x @32k, 1.36x @98k faster —
while reading a 1.88x smaller cache (~10.9 KB/token). vs the mlx_lm unfused
quantized path it replaces: 2.7x @8k, 3.4x @32k, 5.4x @98k. Numerics:
max |err| vs a dequantize->fp32 reference is at output-dtype rounding level
(fp16 ~4e-4 rel, bf16 ~3e-3 rel), same acceptance class as mlx's own fused
kernels.

`install()` patches the QuantizedKVCache branch of
`mlx_lm.models.base.scaled_dot_product_attention` to use the fused kernels
for S==1 decode steps on exactly this shape (D=256, GQA=8, bits=8, gs=64,
no/causal mask), reading the cache's FULL padded buffers with a runtime
valid-length — the sliced views the stock path uses would force a
contiguity copy per call. Everything else falls through to stock. Opt out:
CHAD_NO_QSDPA=1.

Scope note: chad's engine still gates kv_bits OFF the rewind/interruptible-
prefill paths (engine.py), so this kernel lands ahead of that glue; it also
serves `--backend openai` setups driving mlx_lm.server with --kv-bits.
"""

from typing import Any

from . import config
from .diag import log

_HEADER = """
#include <metal_simdgroup>
#include <metal_math>
using namespace metal;
constant constexpr float FIN_MIN = -3.402823466e38f;
"""

# Pass 1. Threadgroups (HKV, B, blocks) x (32 lanes, 8 simdgroups); one
# simdgroup per q head of the kv head. Chunks of CH=8 positions are
# dequantized cooperatively (simdgroup j -> position i0+j) into double-
# buffered threadgroup memory; the next chunk's packed uint2/scale loads are
# issued before the current chunk's math so device latency overlaps ALU.
# Buffers are the cache's FULL padded allocations; params carries the valid
# length N (runtime — a template N would recompile as the cache grows).
_P1_SRC = """
  constexpr int BD = 32;
  constexpr int QK = D / BD;           // 8 elements per lane
  constexpr int PW = D / 4;            // 64 packed uint32 per row
  constexpr int GW = D / 64;           // 4 scale groups per row
  constexpr int CH = 8;                // positions per staged chunk
  constexpr int RPS = CH / GQA;        // staged rows per simdgroup (1 @gqa8)

  const int N      = params[0];
  const int NP     = params[1];
  const int blocks = params[2];

  const int kv_head_idx = threadgroup_position_in_grid.x;
  const int batch_idx   = threadgroup_position_in_grid.y;
  const int block_idx   = threadgroup_position_in_grid.z;
  const int num_kv_heads = threadgroups_per_grid.x;
  const int lane = thread_index_in_simdgroup;
  const int sg   = simdgroup_index_in_threadgroup;

  const int num_q_heads = num_kv_heads * GQA;
  const int q_head_idx = kv_head_idx * GQA + sg;
  const int q_batch_head_idx = batch_idx * num_q_heads + q_head_idx;

  const device T* q_ptr = q + q_batch_head_idx * D + lane * QK;
  const size_t kv_row0 = (size_t)(batch_idx * num_kv_heads + kv_head_idx) * NP;

  device float* out_p = partials
      + ((size_t)q_batch_head_idx * blocks + block_idx) * D + lane * QK;
  const size_t sm_off = (size_t)q_batch_head_idx * blocks + block_idx;

  threadgroup half Ksh[2 * CH * D];
  threadgroup half Vsh[2 * CH * D];

  float4 q0, q1;
  for (int j = 0; j < 4; j++) q0[j] = (float)q_ptr[j];
  for (int j = 0; j < 4; j++) q1[j] = (float)q_ptr[4 + j];
  q0 *= scale[0]; q1 *= scale[0];

  float4 o0 = 0.0f, o1 = 0.0f;
  float max_score = FIN_MIN;
  float sum_exp_score = 0.0f;

  const int stride = blocks * CH;

  // prologue: stage chunk 0 into buffer 0 (each simdgroup stages RPS rows)
  for (int rr = 0; rr < RPS; rr++) {
    const int pos = block_idx * CH + sg + rr * GQA;
    if (pos < N) {
      const size_t row = kv_row0 + pos;
      uint2 wa = ((const device uint2*)(kw + row * PW))[lane];
      uint2 xa = ((const device uint2*)(vw + row * PW))[lane];
      float ksc = (float)ks[row * GW + lane / 8];
      float kbi = (float)kb[row * GW + lane / 8];
      float vsc = (float)vs[row * GW + lane / 8];
      float vbi = (float)vb[row * GW + lane / 8];
      const int slot = sg + rr * GQA;
      threadgroup half4* kd = (threadgroup half4*)(Ksh + slot * D + lane * QK);
      threadgroup half4* vd = (threadgroup half4*)(Vsh + slot * D + lane * QK);
      kd[0] = half4(ksc * float4(as_type<uchar4>(wa.x)) + kbi);
      kd[1] = half4(ksc * float4(as_type<uchar4>(wa.y)) + kbi);
      vd[0] = half4(vsc * float4(as_type<uchar4>(xa.x)) + vbi);
      vd[1] = half4(vsc * float4(as_type<uchar4>(xa.y)) + vbi);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  int buf = 0;
  for (int i0 = block_idx * CH; i0 < N; i0 += stride, buf ^= 1) {
    // prefetch the NEXT chunk's packed rows (device loads overlap the math)
    uint2 wa[RPS], xa[RPS];
    float ksc[RPS], kbi[RPS], vsc[RPS], vbi[RPS];
    bool have_next[RPS];
    for (int rr = 0; rr < RPS; rr++) {
      const int npos = i0 + stride + sg + rr * GQA;
      have_next[rr] = npos < N;
      if (have_next[rr]) {
        const size_t row = kv_row0 + npos;
        wa[rr] = ((const device uint2*)(kw + row * PW))[lane];
        xa[rr] = ((const device uint2*)(vw + row * PW))[lane];
        ksc[rr] = (float)ks[row * GW + lane / 8];
        kbi[rr] = (float)kb[row * GW + lane / 8];
        vsc[rr] = (float)vs[row * GW + lane / 8];
        vbi[rr] = (float)vb[row * GW + lane / 8];
      }
    }

    // consume the staged chunk: scores, online softmax, V accumulate
    const threadgroup half* Kc = Ksh + buf * CH * D;
    const threadgroup half* Vc = Vsh + buf * CH * D;
    const int valid = min(CH, N - i0);
    float sc[CH];
    for (int j = 0; j < CH; j++) {
      const threadgroup half4* kr = (const threadgroup half4*)(Kc + j * D + lane * QK);
      float d = dot(q0, float4(kr[0])) + dot(q1, float4(kr[1]));
      sc[j] = simd_sum(d);
    }
    float cmax = FIN_MIN;
    for (int j = 0; j < valid; j++) cmax = max(cmax, sc[j]);
    float new_max = max(max_score, cmax);
    float factor = fast::exp(max_score - new_max);
    float e[CH];
    float esum = 0.0f;
    for (int j = 0; j < CH; j++) {
      e[j] = (j < valid) ? fast::exp(sc[j] - new_max) : 0.0f;
      esum += e[j];
    }
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + esum;
    o0 *= factor; o1 *= factor;
    for (int j = 0; j < CH; j++) {
      const threadgroup half4* vr = (const threadgroup half4*)(Vc + j * D + lane * QK);
      o0 += e[j] * float4(vr[0]);
      o1 += e[j] * float4(vr[1]);
    }

    // dequant the prefetched rows into the other buffer
    for (int rr = 0; rr < RPS; rr++) {
      if (have_next[rr]) {
        const int slot = sg + rr * GQA;
        threadgroup half4* kd = (threadgroup half4*)(Ksh + (buf ^ 1) * CH * D + slot * D + lane * QK);
        threadgroup half4* vd = (threadgroup half4*)(Vsh + (buf ^ 1) * CH * D + slot * D + lane * QK);
        kd[0] = half4(ksc[rr] * float4(as_type<uchar4>(wa[rr].x)) + kbi[rr]);
        kd[1] = half4(ksc[rr] * float4(as_type<uchar4>(wa[rr].y)) + kbi[rr]);
        vd[0] = half4(vsc[rr] * float4(as_type<uchar4>(xa[rr].x)) + vbi[rr]);
        vd[1] = half4(vsc[rr] * float4(as_type<uchar4>(xa[rr].y)) + vbi[rr]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0) {
    sums[sm_off] = sum_exp_score;
    maxs[sm_off] = max_score;
  }
  for (int j = 0; j < 4; j++) out_p[j] = o0[j];
  for (int j = 0; j < 4; j++) out_p[4 + j] = o1[j];
"""

# Pass 2: identical math to mlx's sdpa_vector_2pass_2 (fp32 partials in,
# combine `blocks` per-block (max, sum, partial) triples per head).
_P2_SRC = """
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int EPT = D / BD;          // 8

  const int blocks = params[2];

  const int head_idx = threadgroup_position_in_grid.x;
  const int lane = thread_index_in_simdgroup;
  const int sg   = simdgroup_index_in_threadgroup;

  const device float* part_p = partials
      + (size_t)head_idx * blocks * D + sg * D + lane * EPT;
  const device float* sums_p = sums + (size_t)head_idx * blocks;
  const device float* maxs_p = maxs + (size_t)head_idx * blocks;
  device T* out_p = out + (size_t)head_idx * D + sg * EPT;

  float o[EPT] = {0.0f};
  threadgroup float outputs[BN * BD];

  float max_score = FIN_MIN;
  for (int b = 0; b < blocks / BN; ++b) {
    max_score = max(max_score, maxs_p[lane + BN * b]);
  }
  max_score = simd_max(max_score);

  float sum_exp_score = 0.0f;
  for (int b = 0; b < blocks / BN; ++b) {
    sum_exp_score += fast::exp(maxs_p[lane + BN * b] - max_score) * sums_p[lane + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);

  for (int b = 0; b < blocks / BN; ++b) {
    float factor = fast::exp(maxs_p[sg + BN * b] - max_score);
    for (int i = 0; i < EPT; i++) {
      o[i] += factor * part_p[BN * D * b + i];
    }
  }

  for (int i = 0; i < EPT; i++) {
    outputs[lane * BD + sg] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[sg * BD + lane]);
    o[i] = sum_exp_score == 0.0f ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0) {
    for (int i = 0; i < EPT; i++) {
      out_p[i] = static_cast<T>(o[i]);
    }
  }
"""

_GQAS = (4, 8)   # 35B is 16q/2kv (gqa 8); 9B is 16q/4kv (gqa 4)
_D = 256


def covers(head_dim: int, gqa: int) -> bool:
    """True iff the fused decode kernel handles this attention shape (the
    engine uses this to decide whether quantized-KV should default ON)."""
    return head_dim == _D and gqa in _GQAS

_p1 = None
_p2 = None


def _kernels():
    global _p1, _p2
    if _p1 is None:
        import mlx.core as mx
        _p1 = mx.fast.metal_kernel(
            name="chad_qsdpa_p1",
            input_names=["q", "kw", "ks", "kb", "vw", "vs", "vb", "scale", "params"],
            output_names=["partials", "sums", "maxs"],
            header=_HEADER,
            source=_P1_SRC,
        )
        _p2 = mx.fast.metal_kernel(
            name="chad_qsdpa_p2",
            input_names=["partials", "sums", "maxs", "params"],
            output_names=["out"],
            header=_HEADER,
            source=_P2_SRC,
        )
    return _p1, _p2


def _pick_blocks(n: int) -> int:
    """Stock sdpa_vector_2pass block table for 's'-class devices (M4 family)
    at n_simds=8; also keeps blocks a multiple of 32 for pass 2."""
    if n <= 8192:
        return 128
    if n <= 32768:
        return 256
    if n <= 65536:
        return 512
    return 1024


def qsdpa(q: Any, k_quant: tuple, v_quant: tuple, scale: float, n: int) -> Any:
    """Fused decode attention over full padded QuantizedKVCache buffers.

    q: (B, HQ, 1, 256); k_quant/v_quant: (weights, scales, biases) with seq
    length NP >= n; n: valid positions. Caller guarantees the shape/quant
    contract (see `_eligible`).
    """
    import mlx.core as mx

    kw, ks, kb = k_quant
    vw, vs, vb = v_quant
    B, HQ, S, D = q.shape
    HKV, NP = kw.shape[1], kw.shape[2]
    gqa = HQ // HKV
    blocks = _pick_blocks(n)
    params = mx.array([n, NP, blocks], dtype=mx.int32)
    scale_arr = mx.array([scale], dtype=mx.float32)
    p1, p2 = _kernels()

    partials, sums, maxs = p1(
        inputs=[q, kw, ks, kb, vw, vs, vb, scale_arr, params],
        template=[("T", q.dtype), ("D", D), ("GQA", gqa)],
        grid=(32 * HKV, gqa * B, blocks),
        threadgroup=(32, gqa, 1),
        output_shapes=[(B, HQ, S, blocks, D), (B, HQ, S, blocks),
                       (B, HQ, S, blocks)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    (out,) = p2(
        inputs=[partials, sums, maxs, params],
        template=[("T", q.dtype), ("D", D)],
        grid=(1024 * B * HQ * S, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(B, HQ, S, D)],
        output_dtypes=[q.dtype],
    )
    return out


def _eligible(q: Any, cache: Any, mask: Any) -> bool:
    """True iff this call is exactly the validated decode shape: S==1,
    D==256, GQA==8, 8-bit group-64 quantized cache, no restricting mask."""
    import mlx.core as mx

    if mask is not None and not (isinstance(mask, str) and mask == "causal"):
        return False
    if q.ndim != 4 or q.shape[2] != 1 or q.shape[3] != _D:
        return False
    if q.dtype not in (mx.float16, mx.bfloat16):
        return False
    if cache.bits != 8 or cache.group_size != 64:
        return False
    keys = cache.keys
    if keys is None or cache.offset <= 0:
        return False
    kw, ks, _ = keys
    if kw.ndim != 4 or kw.shape[1] == 0 or q.shape[1] % kw.shape[1]:
        return False
    if q.shape[1] // kw.shape[1] not in _GQAS:
        return False
    if kw.shape[3] != _D // 4 or ks.dtype != q.dtype:
        return False
    if cache.offset > kw.shape[2]:  # cache invariant violated; stay stock
        return False
    return True


def install() -> bool:
    """Patch the QuantizedKVCache branch of mlx_lm's attention helper to use
    the fused kernels on eligible decode steps. Safe no-op on failure."""
    if config.flag("CHAD_NO_QSDPA"):
        return False
    try:
        import mlx.core as mx  # noqa: F401
        from mlx_lm.models import base as lm_base
        from mlx_lm.models.cache import QuantizedKVCache
    except ImportError:
        return False
    if getattr(lm_base.scaled_dot_product_attention, "_chad_qsdpa", False):
        return True  # already installed

    stock = lm_base.scaled_dot_product_attention

    def patched(queries, keys, values, cache, scale, mask=None, sinks=None):
        if sinks is None and type(cache) is QuantizedKVCache:
            if _eligible(queries, cache, mask):
                try:
                    return qsdpa(queries, cache.keys, cache.values, scale,
                                 cache.offset)
                except Exception as e:  # noqa: BLE001 — perf path: never break decode
                    log.warning("qsdpa fused kernel failed (%s); stock path", e)
            elif queries.ndim == 4 and queries.shape[2] > 1:
                # Prefill (S>1) over a quantized cache: dequantize this
                # layer's K/V and run the FUSED fp16/bf16 kernel (steel at
                # hd-256 via the patched wheel) instead of mlx_lm's unfused
                # qmm+softmax+qmm — which materializes an (H, S, N) score
                # slab (~2 GB at chunk 2048 x 32k ctx) and is several times
                # slower. The dequantized transient is N*D*4 bytes per
                # kv-head (~67 MB at 32k), freed at chunk end.
                try:
                    import mlx.core as mx
                    kd = mx.dequantize(*keys, group_size=cache.group_size,
                                       bits=cache.bits)
                    vd = mx.dequantize(*values, group_size=cache.group_size,
                                       bits=cache.bits)
                    return mx.fast.scaled_dot_product_attention(
                        queries, kd, vd, scale=scale, mask=mask)
                except Exception as e:  # noqa: BLE001 — same contract as above
                    log.warning("qsdpa prefill dequant failed (%s); stock "
                                "path", e)
        return stock(queries, keys, values, cache=cache, scale=scale,
                     mask=mask, sinks=sinks)

    patched._chad_qsdpa = True  # type: ignore[attr-defined]
    lm_base.scaled_dot_product_attention = patched
    # models import the helper by name at module load; rebind any that did.
    import sys
    for mod_name, mod in list(sys.modules.items()):
        if (mod_name.startswith("mlx_lm.models.")
                and getattr(mod, "scaled_dot_product_attention", None) is stock):
            setattr(mod, "scaled_dot_product_attention", patched)
    log.info("QSDPA installed: fused quantized-KV decode attention")
    return True
