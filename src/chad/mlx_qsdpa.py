"""Fused quantized-KV decode attention.

mlx_lm's quantized-KV attention (`kv_bits=8`) is manually unfused — two
`quantized_matmul`s around a materialized softmax — and measured 2.3-2.6x
SLOWER than fused fp16 attention despite reading half the bytes (measured
53->38 tok/s @32k). That made cache quantization a pure loss, so chad shipped
it off and the governor's ctx_limit paid full fp16 KV rates. With the kernels
below it is the DEFAULT (see Engine._resolve_kv_bits): a win on decode time and
on the context the governor can afford, rather than a trade between them.

This module JIT-compiles (mx.fast.metal_kernel — no wheel rebuild) a fused
online-softmax scan over the 8-bit group-64 affine QuantizedKVCache layout.
Pass-1 schedules of the same math share one pass 2:

- **per-head** (`_P1_SRC`): structured like mlx's sdpa_vector_2pass with one
  addition — each threadgroup cooperatively dequantizes a CHUNK of 8 positions
  into threadgroup memory ONCE, double-buffered so the next chunk's packed loads
  overlap the current chunk's math, and the GQA q-head simdgroups consume cheap
  staged rows. (A naive port that dequantized per-head was ALU-bound and lost to
  fp16; sharing the dequant is what flips it.) This is the S==1 decode path.

- **simdgroup_matrix retile, wide** (`_P1_SGM_WIDE_SRC` / `_P1_SGM_WIDE1R_SRC`):
  for S>1 shapes — speculative verification and prefill. A simdgroup owns a
  (position-stream, output-dim-quarter) instead of a q head, so each staged row is
  consumed once instead of GQA times and the products become matrix ops. The 8 tile
  rows are VIRTUAL (head, position) rows, which is what frees it from the head-count
  constraint its S==1 predecessor had.

A third schedule used to serve S==1 with the same retile over REAL head rows. It
needed exactly 8 q heads per kv head to fill its fixed 8x8 score tile and was worth
1.26-1.30x at >=8k on the gqa-8 Ornith 35B shape; 2.0.0 removed it along with that
model, since the shipped checkpoint is gqa-6 and could never dispatch it.

Profiling that motivated the retile: ablating the math phase left the per-head
kernel streaming at 331 GB/s, at the measured roofline, with 68% of its runtime
in the math. The cost was GQA reuse through threadgroup memory (~8 KB/position
against ~500 B/position of device traffic), not the fetch.

Measured when the kernel was built, on what was then the shipped shape
(q 16h x 256d, kv 2h, GQA-8, bf16, M4 Pro) — the ratios are the reason this
exists, not a claim about the current model:
vs stock fused fp16 sdpa_vector: 1.17x @8k, 1.31x @32k, 1.36x @98k faster —
while reading a 1.88x smaller cache (~10.9 KB/token). vs the mlx_lm unfused
quantized path it replaces: 2.7x @8k, 3.4x @32k, 5.4x @98k. Numerics:
max |err| vs a dequantize->fp32 reference is at output-dtype rounding level
(fp16 ~4e-4 rel, bf16 ~3e-3 rel), same acceptance class as mlx's own fused
kernels.

`install()` patches the QuantizedKVCache branch of
`mlx_lm.models.base.scaled_dot_product_attention` to use the fused kernels on
the shapes `covers()` accepts (D=256, GQA 4/6/8 — the shipped model is GQA-6 —
bits=8, gs=64, no/causal mask): S==1 decode steps, and verify steps up to
`_wide_s_max`. It reads the cache's FULL padded buffers with a runtime
valid-length — the sliced views the stock path uses would force a contiguity
copy per call. Wider steps (deep speculation, prefill chunks) take the same
patch's other branch, which dequantizes that layer's K/V and runs stock fused
attention; past `_wide_s_max` that is measured the faster of the two.
Everything else falls through to stock. Opt out: CHAD_NO_QSDPA=1.

Scope note: the engine builds the cache quantized from the start, so these
kernels serve normal decode, MTP verification and prefill alike. The one path
that still requires an fp16 cache is opt-in wide prompt-lookup decoding
(CHAD_USE_PLD). This also serves `chad serve` setups exposing the in-process
engine with --kv-bits.
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
  constexpr int RPS = (8 / GQA) > 0 ? (8 / GQA) : 1;  // staged rows per simdgroup
  constexpr int CH = RPS * GQA;        // positions per staged chunk: must be a
                                       // multiple of GQA or the staging loop
                                       // (slot = sg + rr*GQA) leaves slots
                                       // unwritten while the math consumes all
                                       // CH of them (8 @gqa4/8, 6 @gqa6)

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

# Pass 1, wide-S form (S in 2..6): the MTP verify forward's shape. Same
# staging/double-buffering as _P1_SRC; a simdgroup still owns one q head but
# now carries that head's S query rows (staged pre-scaled into threadgroup
# memory — S x 8 floats of extra registers per lane would spill at S >= 4),
# and consumes each staged K/V chunk against all S rows, so the quantized
# cache is read from device ONCE per verify instead of per row — the whole
# point: the dequantize-then-fused fallback this replaces moves ~5x the bytes.
# Causality inside the tail is a per-row valid length (row s of the S new
# positions may attend to N - S + 1 + s positions); no mask tensor exists.
# Output contract is pass 2's existing S-carrying one: partials
# (B, HQ, S, blocks, D), sums/maxs (B, HQ, S, blocks).
_P1_WIDE_SRC = """
  constexpr int BD = 32;
  constexpr int QK = D / BD;           // 8 elements per lane
  constexpr int PW = D / 4;            // 64 packed uint32 per row
  constexpr int GW = D / 64;           // 4 scale groups per row
  constexpr int RPS = (8 / GQA) > 0 ? (8 / GQA) : 1;
  constexpr int CH = RPS * GQA;

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

  const size_t kv_row0 = (size_t)(batch_idx * num_kv_heads + kv_head_idx) * NP;

  threadgroup half Ksh[2 * CH * D];
  threadgroup half Vsh[2 * CH * D];
  threadgroup half Qsh[GQA * S * D];

  // Stage this simdgroup's S query rows, pre-scaled. Each lane writes and
  // later reads only its own QK slice, so no barrier is needed for Qsh.
  for (int s = 0; s < S; s++) {
    const device T* qp = q + ((size_t)q_batch_head_idx * S + s) * D + lane * QK;
    threadgroup half* qd = Qsh + (sg * S + s) * D + lane * QK;
    for (int j = 0; j < QK; j++) qd[j] = half((float)qp[j] * scale[0]);
  }

  float4 o0[S], o1[S];
  float max_score[S], sum_exp_score[S];
  for (int s = 0; s < S; s++) {
    o0[s] = 0.0f; o1[s] = 0.0f;
    max_score[s] = FIN_MIN; sum_exp_score[s] = 0.0f;
  }

  const int stride = blocks * CH;

  // prologue: stage chunk 0 into buffer 0 (identical to _P1_SRC)
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
    // prefetch the NEXT chunk's packed rows (identical to _P1_SRC)
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

    // Consume the staged chunk: position-outer, row-inner. Each K row is
    // read from threadgroup memory ONCE and serves all S dots (independent
    // simd_sums pipeline instead of serializing one row's reduction chain);
    // each V row likewise feeds all S accumulators. The row-outer form
    // re-read every staged row S times and measured SLOWER than S separate
    // S=1 dispatches; this form is what makes one-read wide pay.
    const threadgroup half* Kc = Ksh + buf * CH * D;
    const threadgroup half* Vc = Vsh + buf * CH * D;
    const int chunk_valid = min(CH, N - i0);   // widest row's limit
    float sc[S][CH];
    for (int j = 0; j < chunk_valid; j++) {
      const threadgroup half4* kr =
          (const threadgroup half4*)(Kc + j * D + lane * QK);
      const float4 k0 = float4(kr[0]);
      const float4 k1 = float4(kr[1]);
      float d[S];
      for (int s = 0; s < S; s++) {
        const threadgroup half4* qr =
            (const threadgroup half4*)(Qsh + (sg * S + s) * D + lane * QK);
        d[s] = dot(float4(qr[0]), k0) + dot(float4(qr[1]), k1);
      }
      for (int s = 0; s < S; s++) sc[s][j] = simd_sum(d[s]);
    }
    float factor[S];
    for (int s = 0; s < S; s++) {
      const int valid = min(CH, (N - S + 1 + s) - i0);  // per-row causal limit
      factor[s] = 1.0f;
      if (valid <= 0) continue;
      float cmax = FIN_MIN;
      for (int j = 0; j < valid; j++) cmax = max(cmax, sc[s][j]);
      const float new_max = max(max_score[s], cmax);
      factor[s] = fast::exp(max_score[s] - new_max);
      max_score[s] = new_max;
      float esum = 0.0f;
      for (int j = 0; j < CH; j++) {
        const float e = (j < valid) ? fast::exp(sc[s][j] - new_max) : 0.0f;
        sc[s][j] = e;                        // sc becomes the exp weights
        esum += e;
      }
      sum_exp_score[s] = sum_exp_score[s] * factor[s] + esum;
      o0[s] *= factor[s]; o1[s] *= factor[s];
    }
    for (int j = 0; j < chunk_valid; j++) {
      const threadgroup half4* vr =
          (const threadgroup half4*)(Vc + j * D + lane * QK);
      const float4 v0 = float4(vr[0]);
      const float4 v1 = float4(vr[1]);
      for (int s = 0; s < S; s++) {
        const int valid = min(CH, (N - S + 1 + s) - i0);
        if (j < valid) {
          o0[s] += sc[s][j] * v0;
          o1[s] += sc[s][j] * v1;
        }
      }
    }

    // dequant the prefetched rows into the other buffer (identical to _P1_SRC)
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

  for (int s = 0; s < S; s++) {
    const size_t sm_off =
        ((size_t)q_batch_head_idx * S + s) * blocks + block_idx;
    if (lane == 0) {
      sums[sm_off] = sum_exp_score[s];
      maxs[sm_off] = max_score[s];
    }
    device float* out_p = partials + sm_off * D + lane * QK;
    for (int j = 0; j < 4; j++) out_p[j] = o0[s][j];
    for (int j = 0; j < 4; j++) out_p[4 + j] = o1[s][j];
  }
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

_HEADER_SGM = """
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_math>
using namespace metal;
constant constexpr float FIN_MIN = -3.402823466e38f;
"""

# Pass 1, simdgroup_matrix form. Same math and same output contract as _P1_SRC;
# the difference is who owns what.
#
# _P1_SRC maps simdgroup -> q head, so all GQA simdgroups re-read the same staged
# K/V rows: ~8 KB/position of threadgroup traffic against ~500 B/position of
# device traffic. Ablation put 68% of that kernel in the math phase while the
# device side alone ran at 331 GB/s -- i.e. at the streaming roofline, with no
# bandwidth left to recover. The cost was the reuse, not the fetch.
#
# Here a simdgroup owns (position-stream, output-dim-quarter):
#
#     sg -> pg = sg >> 2   which of NPG position streams
#           dq = sg & 3    which 64 of the 256 output dims
#
# so each staged row is consumed once by the simdgroup that staged it, and the
# QK/V products become simdgroup_matrix ops. The quarter split is a register
# budget, not a tiling preference: an earlier revision let one simdgroup own all
# 256 output dims (32 simdgroup_float8x8 accumulators, 64 floats/lane) and spilled
# hard. Measured at n=32768 with the matrix-op count held constant --
# O[8] 0.215 ms, O[12] 0.396, O[16] 0.576, O[32] 1.068 -- so 8 tiles is the cliff
# edge and 64 dims is what fits.
#
# Pass 1, SGM retile for the WIDE (S>1 verify) shape. The per-row dispatch
# (S separate S=1 calls) is math-bound in the per-head kernel: at 32k it
# spends ~0.44 of its 0.63 ms/layer/row in simd_sum dot chains. This kernel
# keeps the retile's proven structure — 8-row score tile, NPG=2 position
# streams x 4 dim-quarters, O[8] accumulators (the measured register cliff)
# — but the 8 tile rows become VIRTUAL (head, tail-position) rows of one kv
# head: row r of tile `rt` is (h = (rt*8+r)/S, s = (rt*8+r)%S), all sharing
# the kv head's K/V but each with its own causal limit in the online
# softmax. The row-tile index rides in grid.y (B * RT threadgroup groups),
# so K/V is read once per TILE rather than once per ROW: at gqa 6, S=3 that
# is ceil(18/8)=3 reads instead of 3 per-row reads with the math phase
# collapsed into simdgroup_matrix ops. Rows past GQA*S are padding: staged
# as zeros, e=0 in the softmax, skipped at emission (the partials layout has
# no pad rows). Output contract is pass 2's S-carrying one with
# VB = blocks * NPG.
_P1_SGM_WIDE_SRC = """
  constexpr int DD  = 256;
  constexpr int PW  = DD / 4;
  constexpr int GW  = DD / 64;
  constexpr int CH  = 8;
  constexpr int SL  = 64;
  constexpr int NT  = SL / 8;
  constexpr int NPG = 2;                 // position streams (8 simdgroups)
  constexpr int VROWS = GQA * S;         // real virtual rows per kv head
  constexpr int RT = (VROWS + 7) / 8;    // row tiles (grid.y carries B * RT)

  const int N      = params[0];
  const int NP     = params[1];
  const int blocks = params[2];

  const int kv_head_idx  = threadgroup_position_in_grid.x;
  const int by           = threadgroup_position_in_grid.y;
  const int batch_idx    = by / RT;
  const int rt           = by % RT;
  const int block_idx    = threadgroup_position_in_grid.z;
  const int num_kv_heads = threadgroups_per_grid.x;
  const int lane = thread_index_in_simdgroup;
  const int sg   = simdgroup_index_in_threadgroup;
  const int pg   = sg >> 2;
  const int dq   = sg & 3;

  const int num_q_heads = num_kv_heads * GQA;
  const size_t kv_row0 = (size_t)(batch_idx * num_kv_heads + kv_head_idx) * NP;

  threadgroup half  Qsh[8 * DD];          // this tile's 8 virtual rows
  threadgroup half  Stg[8 * CH * SL];
  threadgroup float Spart[NPG * 4 * 64];
  threadgroup half  Psh[NPG * 64];
  threadgroup half  Fsh[NPG * 64];
  threadgroup float RunM[NPG * 8];
  threadgroup float RunS[NPG * 8];

  // stage the tile's q rows, pre-scaled; pads stay zero
  const float qscale = scale[0];
  for (int i = sg * 32 + lane; i < 8 * DD; i += 32 * 8) {
    const int r = i / DD, d = i % DD;
    const int rv = rt * 8 + r;
    half val = 0.0h;
    if (rv < VROWS) {
      const int h = rv / S, s_ = rv % S;
      const size_t qi = (((size_t)(batch_idx * num_q_heads
                          + kv_head_idx * GQA + h)) * S + s_) * DD + d;
      val = half(qscale * float(q[qi]));
    }
    Qsh[i] = val;
  }
  if (sg < NPG && lane < 8) {
    RunM[sg * 8 + lane] = FIN_MIN;
    RunS[sg * 8 + lane] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup half*  Sl = Stg + sg * CH * SL;
  threadgroup float* Sp = Spart + (pg * 4 + dq) * 64;
  threadgroup half*  Pc = Psh + pg * 64;
  threadgroup half*  Fc = Fsh + pg * 64;
  threadgroup float* Rm = RunM + pg * 8;
  threadgroup float* Rs = RunS + pg * 8;

  simdgroup_float8x8 O[NT];
  for (int t = 0; t < NT; t++) O[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  const int VB = blocks * NPG;
  const int vblk = block_idx * NPG + pg;
  const int total_chunks = (N + CH - 1) / CH;
  const int iters = (total_chunks + VB - 1) / VB;

  for (int it = 0; it < iters; it++) {
    const int i0 = (vblk + it * VB) * CH;
    const int valid = (i0 < N) ? min(CH, N - i0) : 0;

    {
      const int pl = lane >> 2, doff = (lane & 3) * 16;
      threadgroup half4* dst = (threadgroup half4*)(Sl + pl * SL + doff);
      if (pl < valid) {
        const size_t row = kv_row0 + i0 + pl;
        const device uint32_t* src = kw + row * PW + dq * 16 + (lane & 3) * 4;
        const float sc = float(ks[row * GW + dq]);
        const float bi = float(kb[row * GW + dq]);
        for (int u = 0; u < 4; u++) dst[u] = half4(sc * float4(as_type<uchar4>(src[u])) + bi);
      } else {
        for (int u = 0; u < 4; u++) dst[u] = half4(0.0h);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 St = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int j = 0; j < NT; j++) {
      simdgroup_half8x8 A, Bm;
      simdgroup_load(A, Qsh + dq * SL + j * 8, DD);
      simdgroup_load(Bm, Sl + j * 8, SL, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(St, A, Bm, St);
    }
    simdgroup_store(St, Sp, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // dq==0 owns the stream's online softmax; lane = tile row, with a
    // PER-ROW causal limit from the row's tail position s
    if (dq == 0 && lane < 8) {
      const int rv = rt * 8 + lane;
      const bool live = rv < VROWS;
      const int s_ = live ? (rv % S) : 0;
      const int lim = live ? min(valid, (N - S + 1 + s_) - i0) : 0;
      const threadgroup float* q0 = Spart + pg * 4 * 64;
      float s[8];
      float cmax = FIN_MIN;
      for (int p = 0; p < 8; p++) {
        float acc = q0[lane * 8 + p] + q0[64 + lane * 8 + p]
                  + q0[128 + lane * 8 + p] + q0[192 + lane * 8 + p];
        s[p] = acc;
        if (p < lim) cmax = max(cmax, acc);
      }
      const float run_max = Rm[lane];
      const float new_max = max(run_max, cmax);
      const float factor = (run_max == FIN_MIN) ? 0.0f : fast::exp(run_max - new_max);
      float esum = 0.0f;
      for (int p = 0; p < 8; p++) {
        const float e = (p < lim && new_max > FIN_MIN) ? fast::exp(s[p] - new_max) : 0.0f;
        Pc[lane * 8 + p] = half(e);
        esum += e;
        Fc[lane * 8 + p] = half(p == lane ? factor : 0.0f);
      }
      Rs[lane] = Rs[lane] * factor + esum;
      Rm[lane] = new_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
      simdgroup_half8x8 F;
      simdgroup_load(F, Fc, 8);
      for (int t = 0; t < NT; t++) {
        simdgroup_float8x8 tmp;
        simdgroup_multiply(tmp, F, O[t]);
        O[t] = tmp;
      }
    }

    {
      const int pl = lane >> 2, doff = (lane & 3) * 16;
      threadgroup half4* dst = (threadgroup half4*)(Sl + pl * SL + doff);
      if (pl < valid) {
        const size_t row = kv_row0 + i0 + pl;
        const device uint32_t* src = vw + row * PW + dq * 16 + (lane & 3) * 4;
        const float sc = float(vs[row * GW + dq]);
        const float bi = float(vb[row * GW + dq]);
        for (int u = 0; u < 4; u++) dst[u] = half4(sc * float4(as_type<uchar4>(src[u])) + bi);
      } else {
        for (int u = 0; u < 4; u++) dst[u] = half4(0.0h);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8 Pm;
    simdgroup_load(Pm, Pc, 8);
    for (int t = 0; t < NT; t++) {
      simdgroup_half8x8 Vm;
      simdgroup_load(Vm, Sl + t * 8, SL);
      simdgroup_multiply_accumulate(O[t], Pm, Vm, O[t]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // emit: row lane of this tile -> partials row (bhq * S + s); pads skipped
  for (int t = 0; t < NT; t++) {
    simdgroup_store(O[t], Sp, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 8) {
      const int rv = rt * 8 + lane;
      if (rv < VROWS) {
        const int h = rv / S, s_ = rv % S;
        const size_t hq = ((size_t)(batch_idx * num_q_heads
                           + kv_head_idx * GQA + h)) * S + s_;
        device float* dst = partials + (hq * VB + vblk) * DD + dq * SL + t * 8;
        for (int d = 0; d < 8; d++) dst[d] = Sp[lane * 8 + d];
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (dq == 0 && lane < 8) {
    const int rv = rt * 8 + lane;
    if (rv < VROWS) {
      const int h = rv / S, s_ = rv % S;
      const size_t hq = ((size_t)(batch_idx * num_q_heads
                         + kv_head_idx * GQA + h)) * S + s_;
      sums[hq * VB + vblk] = Rs[lane];
      maxs[hq * VB + vblk] = Rm[lane];
    }
  }
"""

# Pass 1, ONE-READ SGM retile for the wide shape: _P1_SGM_WIDE_SRC reads
# K/V once per 8-row TILE (RT reads total); this one reads them ONCE, full
# stop, by keeping all RT row-tiles resident in one threadgroup. The
# register budget forces SL 64 -> 32 (O tiles = RT * SL/8 = 12 at S<=4,
# inside the measured O[12] envelope, vs 24 at SL=64 which spills): 8
# simdgroups become dim-EIGHTHS of one position stream, and each staged
# chunk is consumed by RT sequential score/accumulate tile passes. VB =
# blocks (NPG=1).
_P1_SGM_WIDE1R_SRC = """
  constexpr int DD  = 256;
  constexpr int PW  = DD / 4;
  constexpr int GW  = DD / 64;
  constexpr int CH  = 8;
  constexpr int SL  = 32;                // dims per simdgroup (eighth)
  constexpr int NT  = SL / 8;            // 4 accumulator tiles per row tile
  constexpr int VROWS = GQA * S;
  constexpr int RT = (VROWS + 7) / 8;    // row tiles, all resident here

  const int N      = params[0];
  const int NP     = params[1];
  const int blocks = params[2];

  const int kv_head_idx  = threadgroup_position_in_grid.x;
  const int batch_idx    = threadgroup_position_in_grid.y;
  const int block_idx    = threadgroup_position_in_grid.z;
  const int num_kv_heads = threadgroups_per_grid.x;
  const int lane = thread_index_in_simdgroup;
  const int sg   = simdgroup_index_in_threadgroup;   // dim-eighth 0..7

  const int num_q_heads = num_kv_heads * GQA;
  const size_t kv_row0 = (size_t)(batch_idx * num_kv_heads + kv_head_idx) * NP;

  threadgroup half  Qsh[RT * 8 * DD];     // all virtual rows, pre-scaled
  threadgroup half  Stg[8 * CH * SL];     // one chunk: 8 pos x 256 dims
  threadgroup float Spart[8 * 64];        // 8 dim-eighth partial tiles
  threadgroup half  Psh[RT * 64];
  threadgroup half  Fsh[RT * 64];
  threadgroup float RunM[RT * 8];
  threadgroup float RunS[RT * 8];

  const float qscale = scale[0];
  for (int i = sg * 32 + lane; i < RT * 8 * DD; i += 32 * 8) {
    const int r = i / DD, d = i % DD;
    half val = 0.0h;
    if (r < VROWS) {
      const int h = r / S, s_ = r % S;
      const size_t qi = (((size_t)(batch_idx * num_q_heads
                          + kv_head_idx * GQA + h)) * S + s_) * DD + d;
      val = half(qscale * float(q[qi]));
    }
    Qsh[i] = val;
  }
  if (sg == 0 && lane < RT * 8) {
    RunM[lane] = FIN_MIN;
    RunS[lane] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup half*  Sl = Stg + sg * CH * SL;
  threadgroup float* Sp = Spart + sg * 64;

  simdgroup_float8x8 O[RT * NT];
  for (int t = 0; t < RT * NT; t++) O[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  const int total_chunks = (N + CH - 1) / CH;
  const int iters = (total_chunks + blocks - 1) / blocks;

  for (int it = 0; it < iters; it++) {
    const int i0 = (block_idx + it * blocks) * CH;
    const int valid = (i0 < N) ? min(CH, N - i0) : 0;

    // stage this eighth's K: 8 positions x 32 dims (8 halfs per lane)
    {
      const int pl = lane >> 2, doff = (lane & 3) * 8;
      threadgroup half4* dst = (threadgroup half4*)(Sl + pl * SL + doff);
      if (pl < valid) {
        const size_t row = kv_row0 + i0 + pl;
        const device uint32_t* src = kw + row * PW + sg * 8 + (lane & 3) * 2;
        const float sc = float(ks[row * GW + sg / 2]);
        const float bi = float(kb[row * GW + sg / 2]);
        for (int u = 0; u < 2; u++) dst[u] = half4(sc * float4(as_type<uchar4>(src[u])) + bi);
      } else {
        for (int u = 0; u < 2; u++) dst[u] = half4(0.0h);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (int rt = 0; rt < RT; rt++) {
      // partial S(8r x 8p) over this eighth's dims
      simdgroup_float8x8 St = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      for (int j = 0; j < NT; j++) {
        simdgroup_half8x8 A, Bm;
        simdgroup_load(A, Qsh + (rt * 8) * DD + sg * SL + j * 8, DD);
        simdgroup_load(Bm, Sl + j * 8, SL, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(St, A, Bm, St);
      }
      simdgroup_store(St, Sp, 8);
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (sg == 0 && lane < 8) {
        const int rv = rt * 8 + lane;
        const bool live = rv < VROWS;
        const int s_ = live ? (rv % S) : 0;
        const int lim = live ? min(valid, (N - S + 1 + s_) - i0) : 0;
        float s[8];
        float cmax = FIN_MIN;
        for (int p = 0; p < 8; p++) {
          float acc = 0.0f;
          for (int e8 = 0; e8 < 8; e8++) acc += Spart[e8 * 64 + lane * 8 + p];
          s[p] = acc;
          if (p < lim) cmax = max(cmax, acc);
        }
        const float run_max = RunM[rt * 8 + lane];
        const float new_max = max(run_max, cmax);
        const float factor = (run_max == FIN_MIN) ? 0.0f : fast::exp(run_max - new_max);
        float esum = 0.0f;
        for (int p = 0; p < 8; p++) {
          const float e = (p < lim && new_max > FIN_MIN) ? fast::exp(s[p] - new_max) : 0.0f;
          Psh[rt * 64 + lane * 8 + p] = half(e);
          esum += e;
          Fsh[rt * 64 + lane * 8 + p] = half(p == lane ? factor : 0.0f);
        }
        RunS[rt * 8 + lane] = RunS[rt * 8 + lane] * factor + esum;
        RunM[rt * 8 + lane] = new_max;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      {
        simdgroup_half8x8 F;
        simdgroup_load(F, Fsh + rt * 64, 8);
        for (int t = 0; t < NT; t++) {
          simdgroup_float8x8 tmp;
          simdgroup_multiply(tmp, F, O[rt * NT + t]);
          O[rt * NT + t] = tmp;
        }
      }
    }

    // stage this eighth's V over the SAME chunk, then accumulate all tiles
    {
      const int pl = lane >> 2, doff = (lane & 3) * 8;
      threadgroup half4* dst = (threadgroup half4*)(Sl + pl * SL + doff);
      if (pl < valid) {
        const size_t row = kv_row0 + i0 + pl;
        const device uint32_t* src = vw + row * PW + sg * 8 + (lane & 3) * 2;
        const float sc = float(vs[row * GW + sg / 2]);
        const float bi = float(vb[row * GW + sg / 2]);
        for (int u = 0; u < 2; u++) dst[u] = half4(sc * float4(as_type<uchar4>(src[u])) + bi);
      } else {
        for (int u = 0; u < 2; u++) dst[u] = half4(0.0h);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (int rt = 0; rt < RT; rt++) {
      simdgroup_half8x8 Pm;
      simdgroup_load(Pm, Psh + rt * 64, 8);
      for (int t = 0; t < NT; t++) {
        simdgroup_half8x8 Vm;
        simdgroup_load(Vm, Sl + t * 8, SL);
        simdgroup_multiply_accumulate(O[rt * NT + t], Pm, Vm, O[rt * NT + t]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // emit: partials row (bhq * S + s) at dims [sg*32, sg*32+32)
  for (int rt = 0; rt < RT; rt++) {
    for (int t = 0; t < NT; t++) {
      simdgroup_store(O[rt * NT + t], Sp, 8);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      if (lane < 8) {
        const int rv = rt * 8 + lane;
        if (rv < VROWS) {
          const int h = rv / S, s_ = rv % S;
          const size_t hq = ((size_t)(batch_idx * num_q_heads
                             + kv_head_idx * GQA + h)) * S + s_;
          device float* dst = partials + (hq * blocks + block_idx) * DD + sg * SL + t * 8;
          for (int d = 0; d < 8; d++) dst[d] = Sp[lane * 8 + d];
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  if (sg == 0 && lane < RT * 8) {
    const int rv = lane;
    if (rv < VROWS) {
      const int h = rv / S, s_ = rv % S;
      const size_t hq = ((size_t)(batch_idx * num_q_heads
                         + kv_head_idx * GQA + h)) * S + s_;
      sums[hq * blocks + block_idx] = RunS[lane];
      maxs[hq * blocks + block_idx] = RunM[lane];
    }
  }
"""

# Below this many valid positions the retile's extra barriers and its wider
# partials slab (VB = blocks * NPG) cost more than the reuse it saves, and the
# per-head kernel wins. Crossover measured on the gqa-8 shape, interleaved arms:
# n=1024 0.95x, 2048 1.04x, 4096 1.12x, 8192 1.18x, 16384 1.22x, 98304 1.30x.
_SGM_MIN_N = 2048

# The shipped model is 24q/4kv (gqa 6). 4 and 8 stay covered because `--model`
# is a real escape hatch: they cost nothing but a template instantiation, and
# without them another checkpoint silently loses quantized KV entirely.
_GQAS = (4, 6, 8)
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


_p1_wide = None


def _kernel_wide():
    """Pass 1 in wide-S form (the MTP verify shape); pass 2 is shared — its
    grid already enumerates B*HQ*S heads and the partials contract carries S."""
    global _p1_wide
    if _p1_wide is None:
        import mlx.core as mx
        _p1_wide = mx.fast.metal_kernel(
            name="chad_qsdpa_p1_wide",
            input_names=["q", "kw", "ks", "kb", "vw", "vs", "vb", "scale", "params"],
            output_names=["partials", "sums", "maxs"],
            header=_HEADER,
            source=_P1_WIDE_SRC,
        )
    return _p1_wide


# Largest verify width still worth routing through this kernel instead of the
# dequantize-whole-cache fallback in install()'s S>1 branch. Measured on the
# shipped 27B (one model load, interleaved arms, real verify forwards, PyPI mlx
# 0.32.0) as the round-time gain from taking the fused path at that width:
#
#     S      8k ctx    20k ctx    40k ctx
#      3      +0.1%      +0.1%      +0.0%   <- control: fused on both arms
#      6      -0.0%      +0.4%      +0.5%   <- control
#     10      +8.6%     +19.8%     +36.4%
#     12      +7.8%     +18.5%     +33.5%
#     16      +6.3%     +14.3%     +25.7%
#     24      +2.2%      +6.8%     +12.3%
#     32      -0.6%      -0.7%      -1.7%
#
# The fused path wins by more the longer the context, because the fallback
# dequantizes the whole cache once per attention layer while this kernel reads
# it packed. It crosses over just under S=32, where the fp32 partials slab
# (B, HQ, S, VB, D) that pass 1 writes and pass 2 reads back outgrows the
# dequant it saves — so 32 stays on the fallback, and the cap lands exactly on
# the DepthPolicy candidate below it. A width above the cap is not refused, it
# just takes the other path.
_WIDE_S_CAP = 24

# The same crossover for the PER-ROW schedule, which serves S>1 when the SGM
# retile is unavailable (n < _SGM_MIN_N, or CHAD_NO_QSDPA_WIDE_SGM for
# bisection). It replays the S==1 kernel S times, so its cost is linear in S
# where the fallback's is flat, and it crosses over sooner (gqa 6, per-row vs
# fallback): S=6 2.30x/3.08x, S=10 1.52x/1.92x, S=16 1.01x/1.23x, S=24
# 0.73x/0.86x at 8k/32k. 16 is the last width that is never a loss.
_PER_ROW_S_CAP = 16


def _wide_s_max(gqa: int, n: int = _SGM_MIN_N) -> int:
    """Largest S `_eligible` admits to the fused wide path at this GQA and
    context length — i.e. the cap of whichever schedule `qsdpa` will dispatch.

    The DEFAULT S>1 schedules have no S-dependent threadgroup allocation: the
    SGM retile (S>=3, n>=_SGM_MIN_N) stages one 8-row tile per threadgroup and
    rides the row-tile count in grid.y, and the per-row fallback below it just
    dispatches the S==1 kernel S times. Only `_kernel_wide` — reachable solely
    via CHAD_QSDPA_WIDE_KERNEL — stages GQA*S q rows in threadgroup memory, so
    only it needs the 32 KB budget bound. Applying that bound to every path
    capped the default schedule at 6 and quietly dropped every DepthPolicy
    plateau width (S=10..32) onto the dequantize fallback, while
    Engine._warm_verify_widths went on compiling variants for them.
    """
    if config.flag("CHAD_QSDPA_WIDE_KERNEL"):
        ch = max(8 // gqa, 1) * gqa
        budget = 32768 - 4 * ch * _D * 2      # K/V: 2 buffers x CH x D x half, x2 arrays
        return max(0, min(6, budget // (gqa * _D * 2)))
    if n < _SGM_MIN_N or config.flag("CHAD_NO_QSDPA_WIDE_SGM"):
        return _PER_ROW_S_CAP
    return _WIDE_S_CAP


_p1_sgm_wide_1r = None


def _kernel_sgm_wide_1r():
    """Pass 1, one-read SGM retile for S>1 verify shapes; pass 2 shared
    (VB = blocks, S-carrying partials)."""
    global _p1_sgm_wide_1r
    if _p1_sgm_wide_1r is None:
        import mlx.core as mx
        _p1_sgm_wide_1r = mx.fast.metal_kernel(
            name="chad_qsdpa_p1_sgm_wide_1r",
            input_names=["q", "kw", "ks", "kb", "vw", "vs", "vb", "scale", "params"],
            output_names=["partials", "sums", "maxs"],
            header=_HEADER_SGM,
            source=_P1_SGM_WIDE1R_SRC,
        )
    return _p1_sgm_wide_1r


_p1_sgm_wide = None


def _kernel_sgm_wide():
    """Pass 1, SGM retile for S>1 verify shapes; pass 2 is shared (VB =
    blocks * NPG entries, S-carrying partials)."""
    global _p1_sgm_wide
    if _p1_sgm_wide is None:
        import mlx.core as mx
        _p1_sgm_wide = mx.fast.metal_kernel(
            name="chad_qsdpa_p1_sgm_wide",
            input_names=["q", "kw", "ks", "kb", "vw", "vs", "vb", "scale", "params"],
            output_names=["partials", "sums", "maxs"],
            header=_HEADER_SGM,
            source=_P1_SGM_WIDE_SRC,
        )
    return _p1_sgm_wide


def _sgm_wide_ok(n: int, S: int) -> bool:
    """Whether the S>1 simdgroup retile serves this shape. Single source of the
    predicate: `qsdpa` branches on it and `_pick_blocks` has to agree with the
    branch it is sizing."""
    return (S >= 3 and n >= _SGM_MIN_N
            and not config.flag("CHAD_NO_QSDPA_WIDE_SGM")
            and not config.flag("CHAD_QSDPA_WIDE_KERNEL"))


def _uses_rt_split(n: int, S: int) -> bool:
    """...and specifically its RT-split form rather than the one-read form."""
    return _sgm_wide_ok(n, S) and (S > 4 or config.flag("CHAD_QSDPA_WIDE_SGM_RT"))


def _pick_blocks(n: int, gqa: int = 8, S: int = 1) -> int:
    """Split factor for the 2-pass scan. Pass 2 combines `blocks` partials per
    head, so this must stay a multiple of its BN=32.

    The RT-split wide kernel takes a flat 32; every other schedule takes the
    context table below. That split is not about S as such — it is about which
    kernel runs. RT-split hands pass 2 `blocks * 2` entries per head, so its
    fp32 partials slab, (B, HQ, S, VB, D), is twice as deep per block as any
    other schedule's *and* S times as tall, while the grid already carries S
    times the work and has no occupancy left to buy. Sweeping the boundary
    (gqa 6, b32 gain over b64) puts the crossover exactly on the kernel change
    at S=5, in both contexts:

        S           1      2      3      4  |    5      6      8     10     24
        n=16384  -1.8%  -5.7%  -1.6%  +0.2% | +6.1%  +5.3%  +5.1%  +6.0%  +5.7%
        n=32768  -4.7%  -3.2%  -4.0%  -2.3% | +3.0%  +1.1%  +2.3%  +2.0%  +2.5%

    S<=4 is the one-read form (or, at S=2, the per-row replay of the S==1
    kernel), and those want the same wide split single-token decode wants.

    Deliberately NOT mlx's sdpa_vector_2pass table, which this kernel used to
    inherit. That table is tuned for an fp16 cache, which reads twice the bytes
    per position and so needs more blocks to saturate; here the extra blocks buy
    no occupancy (32 blocks x 2 kv heads already fills the GPU) and cost a
    (heads, blocks, 256) fp32 partials slab that pass 1 writes and pass 2 reads
    back. At blocks=256 that slab is a fifth of the kernel's whole memory
    traffic. Measured on the 35B shape, min-of-interleaved-arms, M4 Pro:

        n       blocks 32   stock table   delta
        8192      0.171 ms   0.179 (128)  -4.5%
        32768     0.354 ms   0.369 (256)  -4.1%
        98304     0.990 ms   1.078 (1024) -8.2%

    Fewer than 32 is not available (pass 2's BN), and the isolated curve is
    monotone up to it — but the isolated curve is not the decision criterion
    (see the module maxim), and end-to-end the answer moved once the S==1
    simdgroup retile landed for gqa 8, whose pass-1 partials were cheaper per
    block. Measured end-to-end on the retired models (one load, interleaved
    arms, median-of-4 paired rounds, spread clearing 1.000):

        gqa 8 (35B): 64 is 1.006x+-0.002 @16k, 1.015x+-0.001 @32k; 1.000 @8k;
                     128 is 0.995 @16k — worse than 64 everywhere it was tried.
        gqa 4 (9B):  128 is 1.005x+-0.002 @32k; exactly 1.000 @16k; noise @8k.

    Both tiers widen at n >= 16384: the largest measured-neutral point is 16000
    and the wins start at the next measured context, so the boundary sits just
    above the neutral evidence.

    The gqa-8 half of that evidence is now historical — its retile went with the
    35B, so every S==1 dispatch is the per-head kernel again — but the shipped
    gqa 6 lands in the same branch, and re-measuring it there says the widening
    is still the right call or a tie (S==1, ms): 16384 b32 0.447 / b64 0.439;
    32768 b32 0.758 / b64 0.729 / b128 0.701; 49152 b32 1.036 / b64 1.000.
    64 is best-or-tied at two of three and 4% off b128 at one, which is inside
    the spread that decided the original table — not enough to churn on.
    """
    if _uses_rt_split(n, S):
        return 32
    if n >= 16384:
        return 128 if gqa == 4 else 64
    return 32


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
    blocks = _pick_blocks(n, gqa, S)
    scale_arr = mx.array([scale], dtype=mx.float32)
    _, p2 = _kernels()

    # S==1 always takes the per-head kernel. The retile that once served this shape
    # needed 8 real q head rows to fill its fixed 8x8 score tile, so it was gqa==8
    # only; the shipped model is gqa==6 and could never dispatch it. Its wide (S>1)
    # sibling below carries the retile idea forward on virtual rows, which have no
    # such constraint. Pass 2 is told VB rather than blocks, hence vb_count.
    vb_count = blocks

    params = mx.array([n, NP, blocks], dtype=mx.int32)
    if S > 1:
        if _sgm_wide_ok(n, S):
            # S==2 stays per-row: measured 1.36 vs 1.99 ms at 32k — two
            # row-tiles' barrier traffic isn't paid back by one KV read
            # at only 12 live rows.
            # SGM retile over virtual (head, tail-position) rows: the math
            # phase that makes per-row dispatch 3x the device time collapses
            # into simdgroup_matrix ops. Default is the ONE-READ form (K/V
            # bytes touched once; all row tiles resident, O[RT*4] at SL=32);
            # CHAD_QSDPA_WIDE_SGM_RT forces the RT-split form (one 8-row
            # tile per threadgroup, K/V read once per tile) kept for A/B.
            if _uses_rt_split(n, S):
                # one-read tops out at S=4: RT=5 needs O[20] accumulator
                # tiles, past the measured register cliff (numerics break,
                # not just speed) — S>=5 takes the RT-split form instead.
                # `blocks` is already narrowed for this form (_pick_blocks).
                rt = (gqa * S + 7) // 8
                vb2 = blocks * 2
                partials, sums, maxs = _kernel_sgm_wide()(
                    inputs=[q, kw, ks, kb, vw, vs, vb, scale_arr, params],
                    template=[("T", q.dtype), ("GQA", gqa), ("S", S)],
                    grid=(32 * HKV, 8 * B * rt, blocks),
                    threadgroup=(32, 8, 1),
                    output_shapes=[(B, HQ, S, vb2, D), (B, HQ, S, vb2),
                                   (B, HQ, S, vb2)],
                    output_dtypes=[mx.float32, mx.float32, mx.float32],
                )
            else:
                vb2 = blocks
                partials, sums, maxs = _kernel_sgm_wide_1r()(
                    inputs=[q, kw, ks, kb, vw, vs, vb, scale_arr, params],
                    template=[("T", q.dtype), ("GQA", gqa), ("S", S)],
                    # grid counts THREADS per axis: y needs the full 8-simdgroup
                    # threadgroup height times B, or the tg collapses to (32,1,1)
                    # and only dim-eighth 0 executes
                    grid=(32 * HKV, 8 * B, blocks),
                    threadgroup=(32, 8, 1),
                    output_shapes=[(B, HQ, S, vb2, D), (B, HQ, S, vb2),
                                   (B, HQ, S, vb2)],
                    output_dtypes=[mx.float32, mx.float32, mx.float32],
                )
            (out,) = p2(
                inputs=[partials, sums, maxs,
                        mx.array([n, NP, vb2], dtype=mx.int32)],
                template=[("T", q.dtype), ("D", D)],
                grid=(1024 * B * HQ * S, 1, 1),
                threadgroup=(1024, 1, 1),
                output_shapes=[(B, HQ, S, D)],
                output_dtypes=[q.dtype],
            )
            return out
        if not config.flag("CHAD_QSDPA_WIDE_KERNEL"):
            # Per-row dispatch of the tuned S=1 kernel, each with its own
            # tail-causal valid length. Measured FASTER than the one-read
            # wide kernel (0.7-0.8x) at 8k-49k: the S=1 kernel runs at the
            # streaming roofline, and re-reading the quantized cache S times
            # still beats serializing S rows of math per staged chunk (the
            # wide kernel's math phase is the bottleneck, not the fetch —
            # same lesson as the SGM retile ablation). Both beat the
            # dequantize-whole-cache fallback this path replaces (1.4-1.8x
            # at 32k+). The wide kernel stays behind CHAD_QSDPA_WIDE_KERNEL
            # as the optimization target: a v2 that parallelizes rows across
            # simdgroups instead of serializing them could reclaim the S-fold
            # device read.
            outs = [qsdpa(q[:, :, s : s + 1], k_quant, v_quant, scale,
                          n - S + 1 + s) for s in range(S)]
            return mx.concatenate(outs, axis=2)
        partials, sums, maxs = _kernel_wide()(
            inputs=[q, kw, ks, kb, vw, vs, vb, scale_arr, params],
            template=[("T", q.dtype), ("D", D), ("GQA", gqa), ("S", S)],
            grid=(32 * HKV, gqa * B, blocks),
            threadgroup=(32, gqa, 1),
            output_shapes=[(B, HQ, S, blocks, D), (B, HQ, S, blocks),
                           (B, HQ, S, blocks)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
    else:
        partials, sums, maxs = _kernels()[0](
            inputs=[q, kw, ks, kb, vw, vs, vb, scale_arr, params],
            template=[("T", q.dtype), ("D", D), ("GQA", gqa)],
            grid=(32 * HKV, gqa * B, blocks),
            threadgroup=(32, gqa, 1),
            output_shapes=[(B, HQ, S, blocks, D), (B, HQ, S, blocks),
                           (B, HQ, S, blocks)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
    (out,) = p2(
        inputs=[partials, sums, maxs,
                mx.array([n, NP, vb_count], dtype=mx.int32)],
        template=[("T", q.dtype), ("D", D)],
        grid=(1024 * B * HQ * S, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(B, HQ, S, D)],
        output_dtypes=[q.dtype],
    )
    return out


def _eligible(q: Any, cache: Any, mask: Any) -> bool:
    """True iff this call is a validated decode shape: S==1 (plain decode) or
    S in 2.._wide_s_max(gqa, offset) (the MTP/PLD verify forward), D==256,
    GQA in _GQAS, 8-bit group-64 quantized cache, no restricting mask.

    A width past that cap is not an error — `install()` sends it to the
    dequantize-and-fuse branch, which is the faster of the two up there.

    S>1 additionally requires mask == "causal": the wide kernel implements
    causality as a per-row valid length, which is only correct for the
    standard tail-causal semantics. A None mask at S>1 would mean full
    attention over the tail — not this shape's contract — so it falls through."""
    import mlx.core as mx

    if mask is not None and not (isinstance(mask, str) and mask == "causal"):
        return False
    if q.ndim != 4 or q.shape[3] != _D:
        return False
    S = q.shape[2]
    if S != 1:
        if not (isinstance(mask, str) and mask == "causal"):
            return False
        kw_probe = cache.keys[0] if cache.keys is not None else None
        if kw_probe is None or kw_probe.shape[1] == 0 or q.shape[1] % kw_probe.shape[1]:
            return False
        gqa = q.shape[1] // kw_probe.shape[1]
        if not (2 <= S <= _wide_s_max(gqa, cache.offset)) or cache.offset < S:
            return False
        if config.flag("CHAD_NO_QSDPA_WIDE"):
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


_KERNEL_HEALTHY: bool | None = None   # one-time numeric self-check result (see below)


def kernel_healthy() -> bool:
    """One-time numeric self-check: run the fused kernel on the small synthetic shapes
    that exposed SILENT nan output on M1-class GPUs (GitHub's macos-14 arm64 runners:
    n=3 at gqa 8/4 and n=100 at gqa 8 returned nan while n>=300 was fine — a partial-
    chunk edge this module's runtime try/except can NEVER catch, because the kernel
    doesn't raise, it just poisons the logits) and compare against the dequantize->fp32
    reference. install() refuses the kernel when this fails, so decode falls back to the
    stock quantized path instead of silently generating garbage. Cached per process;
    healthy hardware pays four tiny dispatches at first install(). The kernel itself is
    untouched — the bit-determinism evidence still holds where this passes."""
    global _KERNEL_HEALTHY
    if _KERNEL_HEALTHY is not None:
        return _KERNEL_HEALTHY
    try:
        import mlx.core as mx
        from mlx_lm.models.cache import QuantizedKVCache
    except ImportError:
        _KERNEL_HEALTHY = False
        return False
    scale = _D ** -0.5
    try:
        # The two n=2570 gqa-8 probes are >= _SGM_MIN_N, so they exercise the
        # simdgroup_matrix retile rather than the per-head kernel; without them a
        # GPU could pass this check and still poison every real decode step,
        # since real contexts are never as short as the n=3/n=100 probes.
        # The hq=24/hkv=4 probes are gqa 6 (Qwen3.8-27B): a partial-GQA chunk
        # (CH=6) with its own staging arithmetic, checked at the same edges.
        for hq, hkv, n, dtype, tol in ((16, 2, 3, mx.float16, 4e-3),
                                       (16, 2, 3, mx.bfloat16, 1.6e-2),
                                       (16, 4, 3, mx.float16, 4e-3),
                                       (16, 2, 100, mx.float16, 4e-3),
                                       (16, 2, 2570, mx.float16, 4e-3),
                                       (16, 2, 2570, mx.bfloat16, 1.6e-2),
                                       (24, 4, 3, mx.float16, 4e-3),
                                       (24, 4, 100, mx.bfloat16, 1.6e-2),
                                       (24, 4, 2570, mx.float16, 4e-3)):
            mx.random.seed(7)
            q = mx.random.normal((1, hq, 1, _D)).astype(dtype)
            k = (0.7 * mx.random.normal((1, hkv, n, _D))).astype(dtype)
            v = (0.7 * mx.random.normal((1, hkv, n, _D))).astype(dtype)
            c = QuantizedKVCache(group_size=64, bits=8)
            c.update_and_fetch(k, v)
            assert c.keys is not None and c.values is not None  # set by update_and_fetch
            out = qsdpa(q, c.keys, c.values, scale, n)
            kd = mx.dequantize(*c.keys, group_size=64, bits=8)[..., :n, :]
            vd = mx.dequantize(*c.values, group_size=64, bits=8)[..., :n, :]
            qf = (q.astype(mx.float32) * scale).reshape(1, hkv, hq // hkv, 1, _D)
            scores = qf @ mx.expand_dims(kd.astype(mx.float32), 2).swapaxes(-1, -2)
            p = mx.softmax(scores, axis=-1, precise=True)
            ref = (p @ mx.expand_dims(vd.astype(mx.float32), 2)).reshape(1, hq, 1, _D)
            err = mx.abs(out.astype(mx.float32) - ref).max().item()
            if not err < tol:   # NOT '>=': nan compares False both ways — this catches it
                log.warning("QSDPA self-check FAILED (hkv=%d n=%d %s: err=%s, tol=%s) — "
                            "fused quantized-KV kernel is numerically broken on this "
                            "GPU/toolchain; refusing to install it", hkv, n, dtype, err, tol)
                _KERNEL_HEALTHY = False
                return False
        # Wide-S probes (the MTP verify shape): per-row causal reference —
        # row s of the S tail positions attends to the first n-S+1+s. Probes
        # cover the partial-chunk edges (n near CH), both GQA tiers the wide
        # kernel serves, and both dtypes. A wide kernel that silently NaNs
        # would poison every verify forward, so it gets the same refusal.
        for hq, hkv, n, S, dtype, tol in ((24, 4, 9, 3, mx.float16, 4e-3),
                                          (24, 4, 100, 3, mx.bfloat16, 1.6e-2),
                                          (24, 4, 2570, 3, mx.float16, 4e-3),
                                          (24, 4, 2570, 4, mx.float16, 4e-3),
                                          (24, 4, 2570, 6, mx.bfloat16, 1.6e-2),
                                          (16, 2, 2570, 3, mx.float16, 4e-3),
                                          (16, 4, 2570, 3, mx.float16, 4e-3)):
            mx.random.seed(11)
            q = mx.random.normal((1, hq, S, _D)).astype(dtype)
            k = (0.7 * mx.random.normal((1, hkv, n, _D))).astype(dtype)
            v = (0.7 * mx.random.normal((1, hkv, n, _D))).astype(dtype)
            c = QuantizedKVCache(group_size=64, bits=8)
            c.update_and_fetch(k, v)
            assert c.keys is not None and c.values is not None
            out = qsdpa(q, c.keys, c.values, scale, n)
            kd = mx.dequantize(*c.keys, group_size=64, bits=8)[..., :n, :]
            vd = mx.dequantize(*c.values, group_size=64, bits=8)[..., :n, :]
            qf = (q.astype(mx.float32) * scale).reshape(1, hkv, hq // hkv, S, _D)
            scores = qf @ mx.expand_dims(kd.astype(mx.float32), 2).swapaxes(-1, -2)
            lim = mx.arange(n)[None, :] < (n - S + 1 + mx.arange(S))[:, None]
            scores = mx.where(lim[None, None, None], scores, -mx.inf)
            p = mx.softmax(scores, axis=-1, precise=True)
            ref = (p @ mx.expand_dims(vd.astype(mx.float32), 2)).reshape(1, hq, S, _D)
            err = mx.abs(out.astype(mx.float32) - ref).max().item()
            if not err < tol:
                log.warning("QSDPA wide self-check FAILED (hkv=%d n=%d S=%d %s: "
                            "err=%s, tol=%s) — refusing to install", hkv, n, S,
                            dtype, err, tol)
                _KERNEL_HEALTHY = False
                return False
    except Exception as e:  # noqa: BLE001 — a broken probe means an unusable kernel
        log.warning("QSDPA self-check errored (%s); refusing to install", e)
        _KERNEL_HEALTHY = False
        return False
    _KERNEL_HEALTHY = True
    return True


def warm_widths(widths, hq: int, hkv: int, dtype, n: int = _SGM_MIN_N + 8) -> int:
    """Compile the S-templated pass-1 variants for the verify widths this
    configuration will actually decode at, so none of them compiles inside a
    timed forward.

    Pass 1 for S>=3 carries ("S", S) in its template, so every distinct verify
    width is its own Metal compile the first time it appears: measured 35-120 ms
    on a cold shader cache, ~1-2 ms per process once macOS has archived the
    binary. The absolute wall clock is small, but it lands entirely on the first
    step of the first span -- which is where an A/B reads its delta, and where
    editing a kernel source (which invalidates the archive for every variant)
    puts it back on every run.

    `kernel_healthy` already builds S in {1,2,3,4,6} as a side effect of its
    probes. This closes the rest: S=5 (CHAD_MTP_DRAFT=4) and wide-PLD's 9..32.
    Widths under 3 need nothing -- S==2 and S==1 both dispatch the per-head
    kernel the self-check already built -- and `n` only has to clear
    _SGM_MIN_N, since the variant is keyed on S and not on context length.

    Returns the number of widths warmed. Never raises: a warm-up that fails
    just leaves the compile where it was."""
    try:
        import mlx.core as mx
        from mlx_lm.models.cache import QuantizedKVCache
    except ImportError:
        return 0
    todo = sorted({int(s) for s in widths if int(s) >= 3})
    if not todo or hkv <= 0 or hq % hkv or (hq // hkv) not in _GQAS:
        return 0
    scale = _D ** -0.5
    done = 0
    try:
        mx.random.seed(5)
        k = (0.7 * mx.random.normal((1, hkv, n, _D))).astype(dtype)
        v = (0.7 * mx.random.normal((1, hkv, n, _D))).astype(dtype)
        c = QuantizedKVCache(group_size=64, bits=8)
        c.update_and_fetch(k, v)
        assert c.keys is not None and c.values is not None  # set by update_and_fetch
        for s in todo:
            q = mx.random.normal((1, hq, s, _D)).astype(dtype)
            out = qsdpa(q, c.keys, c.values, scale, n)
            mx.eval(out)
            done += 1
    except Exception as e:  # noqa: BLE001 — a warm-up must never break loading
        log.warning("QSDPA width warm-up stopped after %d/%d (%s)",
                    done, len(todo), e)
    return done


def install() -> bool:
    """Patch the QuantizedKVCache branch of mlx_lm's attention helper to use
    the fused kernels on eligible decode steps. Safe no-op on failure (import
    trouble, CHAD_NO_QSDPA, or a failed numeric self-check — see kernel_healthy)."""
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
    if not kernel_healthy():
        log.warning("QSDPA disabled: numeric self-check failed; stock quantized "
                    "attention stays in place")
        return False

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
                # Too wide for the fused path (prefill chunks, and verify
                # widths past _wide_s_max): dequantize this layer's K/V and
                # run mx.fast SDPA instead of mlx_lm's unfused
                # qmm+softmax+qmm — which materializes an (H, S, N) score
                # slab (~2 GB at chunk 2048 x 32k ctx) and is several times
                # slower. The dequantized transient is N*D*2 bytes per kv-head
                # per array (~67 MB at 32k), freed at chunk end. Cost is flat
                # in S, which is why it wins above the cap and loses below it.
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
