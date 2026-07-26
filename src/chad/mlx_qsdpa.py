"""Fused quantized-KV decode attention (roadmap item 1).

mlx_lm's quantized-KV attention (`kv_bits=8`) is manually unfused — two
`quantized_matmul`s around a materialized softmax — and measured 2.3-2.6x
SLOWER than fused fp16 attention despite reading half the bytes (measured
53->38 tok/s @32k). That made cache quantization a pure loss, so chad ships
with it off and the governor's ctx_limit pays full fp16 KV rates (~20.5
KB/token on the 35B).

This module JIT-compiles (mx.fast.metal_kernel — no wheel rebuild) a fused
online-softmax scan over the 8-bit group-64 affine QuantizedKVCache layout.
There are two pass-1 schedules of the same math, sharing one pass 2:

- **per-head** (`_P1_SGM_SRC`'s predecessor, `_P1_SRC`): structured like mlx's
  sdpa_vector_2pass with one addition — each threadgroup cooperatively
  dequantizes a CHUNK of 8 positions into threadgroup memory ONCE, double-
  buffered so the next chunk's packed loads overlap the current chunk's math,
  and the GQA q-head simdgroups consume cheap staged rows. (A naive port that
  dequantized per-head was ALU-bound and lost to fp16; sharing the dequant is
  what flips it.) Used below `_SGM_MIN_N`, and at gqa 4 always.

- **simdgroup_matrix retile** (`_P1_SGM_SRC`): a simdgroup owns a
  (position-stream, output-dim-quarter) instead of a q head, so each staged row
  is consumed once instead of GQA times and the products become matrix ops.
  Worth 1.26-1.30x over the per-head kernel at >= 8k on the 35B shape. gqa 8
  only — see the gate in `qsdpa()`.

Profiling that motivated the retile: ablating the math phase left the per-head
kernel streaming at 331 GB/s, at the measured roofline, with 68% of its runtime
in the math. The cost was GQA reuse through threadgroup memory (~8 KB/position
against ~500 B/position of device traffic), not the fetch.

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
serves `chad --serve` setups exposing the in-process engine with --kv-bits.
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
# The 4 quarters of a position stream each produce a PARTIAL score tile over
# their own dims; those are summed through threadgroup memory. dq==0 owns the
# group's online-softmax state and publishes the exp weights and the diagonal
# rescale factor. Barriers are threadgroup-wide, so every simdgroup runs the same
# uniform trip count and masks out-of-range work rather than exiting early.
_P1_SGM_SRC = """
  constexpr int DD  = 256;
  constexpr int PW  = DD / 4;    // 64 packed uint32 per row (8-bit)
  constexpr int GW  = DD / 64;   // 4 affine groups per row
  constexpr int CH  = 8;         // positions per chunk
  constexpr int SL  = 64;        // dims owned by one simdgroup
  constexpr int NT  = SL / 8;    // 8 accumulator tiles -- the register budget
  constexpr int NPG = GQA / 4;   // position streams per threadgroup

  const int N      = params[0];
  const int NP     = params[1];
  const int blocks = params[2];

  const int kv_head_idx  = threadgroup_position_in_grid.x;
  const int batch_idx    = threadgroup_position_in_grid.y;
  const int block_idx    = threadgroup_position_in_grid.z;
  const int num_kv_heads = threadgroups_per_grid.x;
  const int lane = thread_index_in_simdgroup;
  const int sg   = simdgroup_index_in_threadgroup;
  const int pg   = sg >> 2;
  const int dq   = sg & 3;

  const int num_q_heads = num_kv_heads * GQA;
  const size_t kv_row0 = (size_t)(batch_idx * num_kv_heads + kv_head_idx) * NP;

  threadgroup half  Qsh[GQA * DD];        // staged once, pre-scaled
  threadgroup half  Stg[GQA * CH * SL];   // per-simdgroup K then V slice
  threadgroup float Spart[NPG * 4 * 64];  // partial score tiles
  threadgroup half  Psh[NPG * 64];        // exp weights (matrix A operand)
  threadgroup half  Fsh[NPG * 64];        // diag(factor) for the O rescale
  threadgroup float RunM[NPG * 8];
  threadgroup float RunS[NPG * 8];

  const float qscale = scale[0];
  for (int i = sg * 32 + lane; i < GQA * DD; i += 32 * GQA) {
    const int h = i / DD, d = i % DD;
    const size_t qi = ((size_t)(batch_idx * num_q_heads + kv_head_idx * GQA + h)) * DD + d;
    Qsh[i] = half(qscale * float(q[qi]));
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

    // stage this quarter's K: 8 positions x 64 dims, 16 halfs per lane
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

    // partial S(8h x 8p) over this quarter's dims
    simdgroup_float8x8 St = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int j = 0; j < NT; j++) {
      simdgroup_half8x8 A, Bm;
      simdgroup_load(A, Qsh + dq * SL + j * 8, DD);              // 8h x 8d
      simdgroup_load(Bm, Sl + j * 8, SL, ulong2(0, 0), true);    // 8d x 8p
      simdgroup_multiply_accumulate(St, A, Bm, St);
    }
    simdgroup_store(St, Sp, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // dq==0 owns the stream's online softmax
    if (dq == 0 && lane < 8) {
      const threadgroup float* q0 = Spart + pg * 4 * 64;
      float s[8];
      float cmax = FIN_MIN;
      for (int p = 0; p < 8; p++) {
        float acc = q0[lane * 8 + p] + q0[64 + lane * 8 + p]
                  + q0[128 + lane * 8 + p] + q0[192 + lane * 8 + p];
        s[p] = acc;
        if (p < valid) cmax = max(cmax, acc);
      }
      const float run_max = Rm[lane];
      const float new_max = max(run_max, cmax);
      const float factor = (run_max == FIN_MIN) ? 0.0f : fast::exp(run_max - new_max);
      float esum = 0.0f;
      for (int p = 0; p < 8; p++) {
        const float e = (p < valid && new_max > FIN_MIN) ? fast::exp(s[p] - new_max) : 0.0f;
        Pc[lane * 8 + p] = half(e);
        esum += e;
        Fc[lane * 8 + p] = half(p == lane ? factor : 0.0f);
      }
      Rs[lane] = Rs[lane] * factor + esum;
      Rm[lane] = new_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // rescale O by diag(factor): simdgroup_float8x8 has no row indexing
    {
      simdgroup_half8x8 F;
      simdgroup_load(F, Fc, 8);
      for (int t = 0; t < NT; t++) {
        simdgroup_float8x8 tmp;
        simdgroup_multiply(tmp, F, O[t]);
        O[t] = tmp;
      }
    }

    // stage this quarter's V, then O(8h x 64d) += P(8h x 8p) @ V(8p x 64d)
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
      simdgroup_load(Vm, Sl + t * 8, SL);                        // 8p x 8d
      simdgroup_multiply_accumulate(O[t], Pm, Vm, O[t]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // emit: this simdgroup owns dims [dq*64, dq*64+64) of virtual block vb
  for (int t = 0; t < NT; t++) {
    simdgroup_store(O[t], Sp, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 8) {
      const size_t hq = (size_t)(batch_idx * num_q_heads + kv_head_idx * GQA + lane);
      device float* dst = partials + (hq * VB + vblk) * DD + dq * SL + t * 8;
      for (int d = 0; d < 8; d++) dst[d] = Sp[lane * 8 + d];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (dq == 0 && lane < 8) {
    const size_t hq = (size_t)(batch_idx * num_q_heads + kv_head_idx * GQA + lane);
    sums[hq * VB + vblk] = Rs[lane];
    maxs[hq * VB + vblk] = Rm[lane];
  }
"""

# Below this many valid positions the retile's extra barriers and its wider
# partials slab (VB = blocks * NPG) cost more than the reuse it saves, and the
# per-head kernel wins. Measured crossover on the 35B shape, interleaved arms:
# n=1024 0.95x, 2048 1.04x, 4096 1.12x, 8192 1.18x, 16384 1.22x, 98304 1.30x.
_SGM_MIN_N = 2048

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


_p1_sgm = None


def _kernel_sgm():
    """Pass 1 in simdgroup_matrix form; pass 2 is shared (same partials contract,
    just VB = blocks * NPG entries instead of blocks)."""
    global _p1_sgm
    if _p1_sgm is None:
        import mlx.core as mx
        _p1_sgm = mx.fast.metal_kernel(
            name="chad_qsdpa_p1_sgm",
            input_names=["q", "kw", "ks", "kb", "vw", "vs", "vb", "scale", "params"],
            output_names=["partials", "sums", "maxs"],
            header=_HEADER_SGM,
            source=_P1_SGM_SRC,
        )
    return _p1_sgm


def _pick_blocks(n: int, gqa: int = 8) -> int:
    """Split factor for the 2-pass scan. Pass 2 combines `blocks` partials per
    head, so this must stay a multiple of its BN=32.

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
    (see the module maxim), and end-to-end the answer moved when the retile
    landed: the flat 32 above was tuned against the PER-HEAD kernel, while at
    gqa 8, n >= _SGM_MIN_N the kernel is now the simdgroup retile, whose pass-1
    partials are cheaper per block. Measured end-to-end on the real models (one
    load, interleaved arms, median-of-4 paired rounds, spread clearing 1.000):

        gqa 8 (35B): 64 is 1.006x+-0.002 @16k, 1.015x+-0.001 @32k; 1.000 @8k;
                     128 is 0.995 @16k — worse than 64 everywhere it was tried.
        gqa 4 (9B):  128 is 1.005x+-0.002 @32k; exactly 1.000 @16k; noise @8k.

    Both tiers widen at n >= 16384: the largest measured-neutral point is 16000
    and the wins start at the next measured context, so the boundary sits just
    above the neutral evidence.
    """
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
    blocks = _pick_blocks(n, gqa)
    scale_arr = mx.array([scale], dtype=mx.float32)
    _, p2 = _kernels()

    # The retile emits one partial per (block, position-stream), so pass 2 has to
    # be told VB rather than blocks. Below _SGM_MIN_N the per-head kernel wins.
    # gqa==8 only: the score tile is a fixed 8x8, so the retile needs 8 real q
    # head rows to fill it. At gqa==4 (the 9B) rows 4-7 would read past Qsh and
    # poison the softmax with NaN, so that tier keeps the per-head kernel.
    use_sgm = (n >= _SGM_MIN_N and gqa == 8
               and not config.flag("CHAD_NO_QSDPA_SGM"))
    npg = (gqa // 4) if use_sgm else 1
    vb_count = blocks * npg

    params = mx.array([n, NP, blocks], dtype=mx.int32)
    if use_sgm:
        partials, sums, maxs = _kernel_sgm()(
            inputs=[q, kw, ks, kb, vw, vs, vb, scale_arr, params],
            template=[("T", q.dtype), ("GQA", gqa)],
            grid=(32 * HKV, gqa * B, blocks),
            threadgroup=(32, gqa, 1),
            output_shapes=[(B, HQ, S, vb_count, D), (B, HQ, S, vb_count),
                           (B, HQ, S, vb_count)],
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
        # The last two are >= _SGM_MIN_N at gqa 8, so they exercise the
        # simdgroup_matrix retile rather than the per-head kernel; without them a
        # GPU could pass this check and still poison every real decode step,
        # since real contexts are never as short as the n=3/n=100 probes.
        for hkv, n, dtype, tol in ((2, 3, mx.float16, 4e-3), (2, 3, mx.bfloat16, 1.6e-2),
                                   (4, 3, mx.float16, 4e-3), (2, 100, mx.float16, 4e-3),
                                   (2, 2570, mx.float16, 4e-3),
                                   (2, 2570, mx.bfloat16, 1.6e-2)):
            mx.random.seed(7)
            q = mx.random.normal((1, 16, 1, _D)).astype(dtype)
            k = (0.7 * mx.random.normal((1, hkv, n, _D))).astype(dtype)
            v = (0.7 * mx.random.normal((1, hkv, n, _D))).astype(dtype)
            c = QuantizedKVCache(group_size=64, bits=8)
            c.update_and_fetch(k, v)
            assert c.keys is not None and c.values is not None  # set by update_and_fetch
            out = qsdpa(q, c.keys, c.values, scale, n)
            kd = mx.dequantize(*c.keys, group_size=64, bits=8)[..., :n, :]
            vd = mx.dequantize(*c.values, group_size=64, bits=8)[..., :n, :]
            qf = (q.astype(mx.float32) * scale).reshape(1, hkv, 16 // hkv, 1, _D)
            scores = qf @ mx.expand_dims(kd.astype(mx.float32), 2).swapaxes(-1, -2)
            p = mx.softmax(scores, axis=-1, precise=True)
            ref = (p @ mx.expand_dims(vd.astype(mx.float32), 2)).reshape(1, 16, 1, _D)
            err = mx.abs(out.astype(mx.float32) - ref).max().item()
            if not err < tol:   # NOT '>=': nan compares False both ways — this catches it
                log.warning("QSDPA self-check FAILED (hkv=%d n=%d %s: err=%s, tol=%s) — "
                            "fused quantized-KV kernel is numerically broken on this "
                            "GPU/toolchain; refusing to install it", hkv, n, dtype, err, tol)
                _KERNEL_HEALTHY = False
                return False
    except Exception as e:  # noqa: BLE001 — a broken probe means an unusable kernel
        log.warning("QSDPA self-check errored (%s); refusing to install", e)
        _KERNEL_HEALTHY = False
        return False
    _KERNEL_HEALTHY = True
    return True


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
