"""Fused MoE decode kernels for the hybrid MoE checkpoint (Ornith 35B class).

The S==1 MoE block is the largest recoverable inefficiency left in decode:
the stock path runs the 8 routed experts through two `gather_qmm`s over tiny
per-expert matrices (1024x2048 at 2-bit / 2048x512 at 3-bit) plus ~8 more dispatches
for routing, the shared expert, and the combine — measured 72-94 GB/s against
a 137 GB/s plateau for well-shaped 2-bit GEMV on the same hardware. This
module JIT-compiles (mx.fast.metal_kernel — no wheel rebuild, same mechanism
as mlx_qsdpa) two kernels that replace the whole post-router block:

  A: gate|up for all 8 routed experts (2-bit branch) AND the shared expert
     (6-bit branch) in ONE dispatch. Row pairs (j, j+512) of the concatenated
     gate|up matrices are computed together so silu(g)*u happens in-register;
     output is h (9, 512) with slot 8 = shared expert. The inner loops are
     mlx's own qmv math verbatim (prescaled x_thread, byte-mask qdot) — that
     inner loop is at its plateau and every attempt to beat it lost, so only
     the SHAPE is changed: one 8-expert dispatch instead of 8 tiny ones.
  B: every down-projection (3-bit routed + 6-bit shared) with routing scores,
     sigmoid(seg) and the residual add folded in; each output block loops the
     experts in-register, so the combine costs no traffic and no atomics.

Routing stays in the op graph, bit-identical to stock (same softmax/top-k/
renormalize ops in the same order), over two bit-exact weight transforms:
  - router `gate` and `shared_expert_gate` concat into one (E+1 -> padded to
    a multiple of 8) 8-bit qmv; per-row math unchanged, zero rows dequantize
    to zero, and padding keeps mlx's fast qmv dispatch.
  - shared expert gate|up concat (6-bit), mirroring the switch_mlp concat.
An algebraic swap (softmax over just the top-8 logits) was measured perf-
neutral and rejected: it perturbed scores for zero win.

Measured on the shipped checkpoint (M4 Pro 24 GB, in-process A/B against the
compiled stock body): 1.28x on the MoE block, +5-7% end-to-end decode. Numerics
are bf16-accumulation class (~1e-2 rel on block output) with expert selection
bit-identical; decode-path perplexity comes out marginally better than stock,
since the combine accumulates in fp32 where the stock graph sums in bf16.

Scope: engages only when every layer matches the exact 35B MoE geometry
(hidden 2048, expert inter 512, top-8, expert gate|up 2-bit / down 3-bit
gs64, shared 6-bit gs64, router/seg 8-bit gs64, norm_topk_prob) — anything else leaves the model
untouched. Prefill (S>1) and batched decode keep the stock op graph over the
fused weights (bit-exact per row). Opt out with CHAD_NO_MOE_FUSED=1.
"""

from typing import Any, Optional

from . import config
from .diag import log

HID = 2048
INTER = 512
NGU = 2 * INTER
TOP_K = 8
GS = 64
RT_BITS = 8   # router|seg concat
SH_BITS = 6   # shared expert gate|up and down
# Launch geometry, single source of truth for warm-up and production calls.
# Mirrors the constexprs baked into _K_A/_K_B (NSG=2; A: RPS=4 over 8 routed
# slots x 64 j-blocks + 64 shared blocks; B: RPS=8 over HID rows).
_TG = (32, 2, 1)
_A_GRID = (32, 2 * (TOP_K * 64 + 64), 1)
_B_GRID = (32, 2 * (HID // 16), 1)

_HEADER = """
#include <metal_simdgroup>
using namespace metal;
"""

# One dispatch, threadgroups (32, 2): tgs [0, 8*64) are the routed experts
# (2-bit), tgs [8*64, 8*64+64) the shared expert (6-bit). Each simdgroup owns
# RPS j's; rows j and j+512 of the concatenated gate|up matrix are reduced in
# the same pass so the silu(g)*u epilogue needs no second dispatch. Inner
# loops are mlx quantized.h qdot/load_vector math, verbatim.
_K_A = """
  constexpr int RPS = 4;
  constexpr int NSG = 2;
  constexpr int K = 2048;
  constexpr int group_size = 64;
  constexpr int blocks_per_slot = 512 / (NSG * RPS);

  const int simd_gid = simdgroup_index_in_threadgroup;
  const int simd_lid = thread_index_in_simdgroup;
  const int tgy = (int)threadgroup_position_in_grid.y;

  if (tgy < 8 * blocks_per_slot) {
    constexpr int PPT = 2;
    constexpr int pack_factor = 16;
    constexpr int bytes_per_pack = 4;
    constexpr int values_per_thread = pack_factor * PPT;
    constexpr int block_size = values_per_thread * 32;
    constexpr int scale_step_per_thread = group_size / values_per_thread;
    constexpr int in_vec_size_w = K * bytes_per_pack / pack_factor;
    constexpr int in_vec_size_g = K / group_size;

    const int slot = tgy / blocks_per_slot;
    const int jbase = (tgy % blocks_per_slot) * (NSG * RPS) + simd_gid * RPS;
    const uint e = inds[slot];

    const device uint8_t* ws = (const device uint8_t*)wq2 +
        (size_t)e * 1024 * in_vec_size_w;
    const device T* se = s2 + (size_t)e * 1024 * in_vec_size_g;
    const device T* be = b2 + (size_t)e * 1024 * in_vec_size_g;

    thread float x_thread[values_per_thread];
    thread float rg[RPS] = {0};
    thread float ru[RPS] = {0};

    const device uint8_t* wl0 = ws + jbase * in_vec_size_w +
        simd_lid * PPT * bytes_per_pack;
    const device T* sl0 = se + jbase * in_vec_size_g +
        simd_lid / scale_step_per_thread;
    const device T* bl0 = be + jbase * in_vec_size_g +
        simd_lid / scale_step_per_thread;
    const device T* xl = x + simd_lid * values_per_thread;
    constexpr size_t uoff = (size_t)512 * in_vec_size_w;
    constexpr size_t uoffg = (size_t)512 * in_vec_size_g;

    for (int k = 0; k < K; k += block_size) {
      float sum = 0.0f;
      for (int i = 0; i < values_per_thread; i += 4) {
        sum += xl[i] + xl[i + 1] + xl[i + 2] + xl[i + 3];
        x_thread[i] = xl[i];
        x_thread[i + 1] = xl[i + 1] / 4.0f;
        x_thread[i + 2] = xl[i + 2] / 16.0f;
        x_thread[i + 3] = xl[i + 3] / 64.0f;
      }
      for (int row = 0; row < RPS; row++) {
        const device uint8_t* wg = wl0 + row * in_vec_size_w;
        const device uint8_t* wu = wg + uoff;
        float ag = 0.0f, au = 0.0f;
        for (int i = 0; i < (values_per_thread / 4); i++) {
          ag += (x_thread[4 * i] * (wg[i] & 0x03) +
                 x_thread[4 * i + 1] * (wg[i] & 0x0c) +
                 x_thread[4 * i + 2] * (wg[i] & 0x30) +
                 x_thread[4 * i + 3] * (wg[i] & 0xc0));
          au += (x_thread[4 * i] * (wu[i] & 0x03) +
                 x_thread[4 * i + 1] * (wu[i] & 0x0c) +
                 x_thread[4 * i + 2] * (wu[i] & 0x30) +
                 x_thread[4 * i + 3] * (wu[i] & 0xc0));
        }
        rg[row] += float(sl0[row * in_vec_size_g]) * ag +
            float(bl0[row * in_vec_size_g]) * sum;
        ru[row] += float(sl0[row * in_vec_size_g + uoffg]) * au +
            float(bl0[row * in_vec_size_g + uoffg]) * sum;
      }
      wl0 += block_size * bytes_per_pack / pack_factor;
      sl0 += block_size / group_size;
      bl0 += block_size / group_size;
      xl += block_size;
    }
    for (int row = 0; row < RPS; row++) {
      float g = simd_sum(rg[row]);
      float u = simd_sum(ru[row]);
      if (simd_lid == 0) {
        h[slot * 512 + jbase + row] =
            static_cast<T>((g / (1.0f + metal::exp(-g))) * u);
      }
    }
  } else {
    constexpr int PPT = 2;
    constexpr int pack_factor = 4;
    constexpr int bytes_per_pack = 3;
    constexpr int values_per_thread = pack_factor * PPT;
    constexpr int block_size = values_per_thread * 32;
    constexpr int scale_step_per_thread = group_size / values_per_thread;
    constexpr int in_vec_size_w = K * bytes_per_pack / pack_factor;
    constexpr int in_vec_size_g = K / group_size;

    const int jbase = (tgy - 8 * blocks_per_slot) * (NSG * RPS) +
        simd_gid * RPS;

    thread float x_thread[values_per_thread];
    thread float rg[RPS] = {0};
    thread float ru[RPS] = {0};

    const device uint8_t* wl0 = (const device uint8_t*)wq6 +
        jbase * in_vec_size_w + simd_lid * PPT * bytes_per_pack;
    const device T* sl0 = s6 + jbase * in_vec_size_g +
        simd_lid / scale_step_per_thread;
    const device T* bl0 = b6 + jbase * in_vec_size_g +
        simd_lid / scale_step_per_thread;
    const device T* xl = x + simd_lid * values_per_thread;
    constexpr size_t uoff = (size_t)512 * in_vec_size_w;
    constexpr size_t uoffg = (size_t)512 * in_vec_size_g;

    for (int k = 0; k < K; k += block_size) {
      float sum = 0.0f;
      for (int i = 0; i < values_per_thread; i += 4) {
        sum += xl[i] + xl[i + 1] + xl[i + 2] + xl[i + 3];
        x_thread[i] = xl[i];
        x_thread[i + 1] = xl[i + 1] / 64.0f;
        x_thread[i + 2] = xl[i + 2] / 16.0f;
        x_thread[i + 3] = xl[i + 3] / 4.0f;
      }
      for (int row = 0; row < RPS; row++) {
        const device uint8_t* wg = wl0 + row * in_vec_size_w;
        const device uint8_t* wu = wg + uoff;
        float ag = 0.0f, au = 0.0f;
        for (int i = 0; i < (values_per_thread / 4); i++) {
          const device uint8_t* p = wg + 3 * i;
          const thread float* xt = x_thread + 4 * i;
          ag += (p[0] & 0x3f) * xt[0];
          ag += (p[0] & 0xc0) * xt[1];
          ag += (p[1] & 0x0f) * (xt[1] * 256.0f);
          ag += (p[1] & 0xf0) * xt[2];
          ag += (p[2] & 0x03) * (xt[2] * 256.0f);
          ag += (p[2] & 0xfc) * xt[3];
          p = wu + 3 * i;
          au += (p[0] & 0x3f) * xt[0];
          au += (p[0] & 0xc0) * xt[1];
          au += (p[1] & 0x0f) * (xt[1] * 256.0f);
          au += (p[1] & 0xf0) * xt[2];
          au += (p[2] & 0x03) * (xt[2] * 256.0f);
          au += (p[2] & 0xfc) * xt[3];
        }
        rg[row] += float(sl0[row * in_vec_size_g]) * ag +
            float(bl0[row * in_vec_size_g]) * sum;
        ru[row] += float(sl0[row * in_vec_size_g + uoffg]) * au +
            float(bl0[row * in_vec_size_g + uoffg]) * sum;
      }
      wl0 += block_size * bytes_per_pack / pack_factor;
      sl0 += block_size / group_size;
      bl0 += block_size / group_size;
      xl += block_size;
    }
    for (int row = 0; row < RPS; row++) {
      float g = simd_sum(rg[row]);
      float u = simd_sum(ru[row]);
      if (simd_lid == 0) {
        h[8 * 512 + jbase + row] =
            static_cast<T>((g / (1.0f + metal::exp(-g))) * u);
      }
    }
  }
"""

# One dispatch, threadgroups (32, 2), each simdgroup owns RPS output rows and
# loops the 8 routed experts (2-bit) then the shared expert (6-bit); scores,
# sigmoid(seg) and the residual are applied in-register so the block's output
# is written exactly once.
_K_B = """
  constexpr int RPS = 8;
  constexpr int NSG = 2;
  constexpr int K = 512;
  constexpr int group_size = 64;
  constexpr int in_vec_size_g = K / group_size;

  const int simd_gid = simdgroup_index_in_threadgroup;
  const int simd_lid = thread_index_in_simdgroup;
  const int rbase = (int)threadgroup_position_in_grid.y * (NSG * RPS) +
      simd_gid * RPS;

  thread float result[RPS] = {0};

  {
    // routed down: 3-bit (the q2_down3 recipe), 8 values per 3 bytes;
    // masks/prescales are mlx quantized.h's bits==3 qdot/load_vector verbatim
    constexpr int values_per_thread = 16;
    constexpr int scale_step_per_thread = group_size / values_per_thread;
    constexpr int in_vec_size_w = K * 3 / 8;

    thread float x_thread[values_per_thread];
    for (int es = 0; es < 8; es++) {
      const uint e = inds[es];
      const float sc_e = sc[es];
      const device T* hl = h + es * K + simd_lid * values_per_thread;
      float sum = 0.0f;
      for (int i = 0; i < values_per_thread; i += 8) {
        sum += hl[i] + hl[i + 1] + hl[i + 2] + hl[i + 3] +
               hl[i + 4] + hl[i + 5] + hl[i + 6] + hl[i + 7];
        x_thread[i] = hl[i];
        x_thread[i + 1] = hl[i + 1] / 8.0f;
        x_thread[i + 2] = hl[i + 2] / 64.0f;
        x_thread[i + 3] = hl[i + 3] / 2.0f;
        x_thread[i + 4] = hl[i + 4] / 16.0f;
        x_thread[i + 5] = hl[i + 5] / 128.0f;
        x_thread[i + 6] = hl[i + 6] / 4.0f;
        x_thread[i + 7] = hl[i + 7] / 32.0f;
      }
      const device uint8_t* ws = (const device uint8_t*)wq2 +
          ((size_t)e * 2048 + rbase) * in_vec_size_w +
          simd_lid * 6;
      const device T* sl = s2 + ((size_t)e * 2048 + rbase) * in_vec_size_g +
          simd_lid / scale_step_per_thread;
      const device T* bl = b2 + ((size_t)e * 2048 + rbase) * in_vec_size_g +
          simd_lid / scale_step_per_thread;
      for (int row = 0; row < RPS; row++) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        float accum = 0.0f;
        for (int i = 0; i < (values_per_thread / 8); i++) {
          const device uint8_t* p = wl + 3 * i;
          const thread float* xt = x_thread + 8 * i;
          accum += (p[0] & 0x07) * xt[0];
          accum += (p[0] & 0x38) * xt[1];
          accum += (p[0] & 0xc0) * xt[2];
          accum += (p[1] & 0x01) * (xt[2] * 256.0f);
          accum += (p[1] & 0x0e) * xt[3];
          accum += (p[1] & 0x70) * xt[4];
          accum += (p[1] & 0x80) * xt[5];
          accum += (p[2] & 0x03) * (xt[5] * 256.0f);
          accum += (p[2] & 0x1c) * xt[6];
          accum += (p[2] & 0xe0) * xt[7];
        }
        result[row] += sc_e * (float(sl[row * in_vec_size_g]) * accum +
                               float(bl[row * in_vec_size_g]) * sum);
      }
    }
  }

  {
    constexpr int pack_factor = 4;
    constexpr int bytes_per_pack = 3;
    constexpr int values_per_thread = 8;
    constexpr int block_size = values_per_thread * 32;
    constexpr int scale_step_per_thread = group_size / values_per_thread;
    constexpr int in_vec_size_w = K * bytes_per_pack / pack_factor;

    const float wsh = 1.0f / (1.0f + metal::exp(-float(seg[0])));
    thread float x_thread[values_per_thread];
    thread float acc6[RPS] = {0};

    const device uint8_t* wl0 = (const device uint8_t*)wq6 +
        rbase * in_vec_size_w + simd_lid * 2 * bytes_per_pack;
    const device T* sl0 = s6 + rbase * in_vec_size_g +
        simd_lid / scale_step_per_thread;
    const device T* bl0 = b6 + rbase * in_vec_size_g +
        simd_lid / scale_step_per_thread;
    const device T* xl = h + 8 * K + simd_lid * values_per_thread;

    for (int k = 0; k < K; k += block_size) {
      float sum = 0.0f;
      for (int i = 0; i < values_per_thread; i += 4) {
        sum += xl[i] + xl[i + 1] + xl[i + 2] + xl[i + 3];
        x_thread[i] = xl[i];
        x_thread[i + 1] = xl[i + 1] / 64.0f;
        x_thread[i + 2] = xl[i + 2] / 16.0f;
        x_thread[i + 3] = xl[i + 3] / 4.0f;
      }
      for (int row = 0; row < RPS; row++) {
        const device uint8_t* wl = wl0 + row * in_vec_size_w;
        float accum = 0.0f;
        for (int i = 0; i < (values_per_thread / 4); i++) {
          const device uint8_t* p = wl + 3 * i;
          const thread float* xt = x_thread + 4 * i;
          accum += (p[0] & 0x3f) * xt[0];
          accum += (p[0] & 0xc0) * xt[1];
          accum += (p[1] & 0x0f) * (xt[1] * 256.0f);
          accum += (p[1] & 0xf0) * xt[2];
          accum += (p[2] & 0x03) * (xt[2] * 256.0f);
          accum += (p[2] & 0xfc) * xt[3];
        }
        acc6[row] += float(sl0[row * in_vec_size_g]) * accum +
            float(bl0[row * in_vec_size_g]) * sum;
      }
      wl0 += block_size * bytes_per_pack / pack_factor;
      sl0 += block_size / group_size;
      bl0 += block_size / group_size;
      xl += block_size;
    }
    for (int row = 0; row < RPS; row++) {
      result[row] += wsh * acc6[row];
    }
  }

  for (int row = 0; row < RPS; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[rbase + row] = static_cast<T>(float(resid[rbase + row]) + result[row]);
    }
  }
"""


def install(model: Any) -> bool:
    """Apply the fused-MoE weight transforms + block patch, in place.

    Called by mlx_fastpath.install() after the switch_mlp gate|up concat.
    Returns True if the transforms were applied (kernels may still fall back
    to the stock-graph body if Metal compilation fails — correctness never
    depends on the kernels). False leaves the model untouched.
    """
    if config.flag("CHAD_NO_MOE_FUSED"):
        return False
    try:
        import mlx.core as mx  # noqa: F401
    except ImportError:
        return False
    try:
        if not _matches(model):
            return False
        # Patch first: it is inert (hasattr guard) until a layer gets _rt_w,
        # and each layer's transform assigns attrs only after every fused
        # array exists — so a failure mid-loop leaves every layer either
        # fully fused (patch serves it) or untouched (stock path serves it).
        _patch_block()
        for layer in model.language_model.model.layers:
            _transform_layer(layer.mlp)
        kernels = _build_kernels(model)
        n = 0
        for layer in model.language_model.model.layers:
            layer.mlp._fused_step = (
                _build_step(layer, kernels) if kernels else None)
            n += layer.mlp._fused_step is not None
        log.info("MOE-FUSED installed: router|seg + shared gate|up concats; "
                 "fused decode kernels on %d layers", n)
        return True
    except Exception as e:  # noqa: BLE001 — perf path must never break loading
        log.warning("moe-fused install failed (%s); running fastpath stock", e)
        return False


def _matches(model) -> bool:
    """True only for the exact geometry the kernels bake in: every layer a
    2048-hidden MoE with 512-inter 2-bit gs64 experts (already concatenated
    by mlx_fastpath), 6-bit gs64 shared expert, 8-bit gs64 router+seg,
    top-8 with normalized routing weights."""
    layers = getattr(getattr(getattr(model, "language_model", None), "model",
                             None), "layers", None)
    if not layers:
        return False
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        sw = getattr(mlp, "switch_mlp", None)
        if mlp is None or sw is None or not hasattr(sw, "_fused_w"):
            return False
        E = mlp.num_experts
        if (mlp.top_k != TOP_K or not mlp.norm_topk_prob or E <= TOP_K
                or getattr(mlp, "sharding_group", None) is not None):
            return False
        if (sw._fused_bits, sw._fused_gs) != (2, GS):
            return False
        if tuple(sw._fused_w.shape) != (E, NGU, HID * 2 // 32):
            return False
        dp = sw.down_proj
        if (dp.bits, dp.group_size) != (3, GS):
            return False
        if tuple(dp.weight.shape) != (E, HID, INTER * 3 // 32):
            return False
        se = mlp.shared_expert
        for m, shp in ((se.gate_proj, (INTER, HID * 6 // 32)),
                       (se.up_proj, (INTER, HID * 6 // 32)),
                       (se.down_proj, (HID, INTER * 6 // 32))):
            if (getattr(m, "bits", None), getattr(m, "group_size", None)) \
                    != (6, GS):
                return False
            if tuple(m.weight.shape) != shp:
                return False
        for m, rows in ((mlp.gate, E), (mlp.shared_expert_gate, 1)):
            if (getattr(m, "bits", None), getattr(m, "group_size", None)) \
                    != (8, GS):
                return False
            if tuple(m.weight.shape) != (rows, HID * 8 // 32):
                return False
        # A checkpoint whose router/shared projections carry an additive bias
        # would pass every shape check and lose the bias in the concat.
        for m in (mlp.gate, mlp.shared_expert_gate,
                  se.gate_proj, se.up_proj, se.down_proj):
            if "bias" in m:
                return False
        # Kernels are template-instantiated per dtype at first CALL, and the
        # warm-up only exercises layer 0 — any intra-layer dtype deviation
        # would surface as an uncaught Metal error mid-generation instead of
        # an install-time fallback. Require one dtype across every tensor the
        # kernels consume.
        dts = {sw._fused_s.dtype, sw._fused_b.dtype,
               dp.scales.dtype, dp.biases.dtype,
               se.gate_proj.scales.dtype, se.gate_proj.biases.dtype,
               se.up_proj.scales.dtype, se.up_proj.biases.dtype,
               se.down_proj.scales.dtype, se.down_proj.biases.dtype,
               mlp.gate.scales.dtype, mlp.gate.biases.dtype,
               mlp.shared_expert_gate.scales.dtype,
               mlp.shared_expert_gate.biases.dtype}
        if len(dts) != 1:
            return False
    import mlx.core as mx
    # bf16 only: that is the regime every parity/NLL gate ran in. fp16
    # checkpoints keep the stock-graph fastpath until separately validated.
    if layers[0].mlp.gate.scales.dtype != mx.bfloat16:
        return False
    return True


def _transform_layer(mlp) -> None:
    """Both weight concats for one layer, atomically: every fused array is
    built and eval'd BEFORE any attribute lands or any original is zeroed, so
    an exception leaves the layer fully stock.

    Shared expert gate|up: one 6-bit qmv instead of two; rows 0..511 gate,
    512..1023 up (bit-exact per row). Router|seg: one 8-bit qmv with rows
    [router 0..E-1 | seg at E | zero pad to a multiple of 8] — zero rows
    dequantize to exactly zero and the padding keeps mlx's fast qmv path."""
    import mlx.core as mx

    se = mlp.shared_expert
    g, u = se.gate_proj, se.up_proj
    shw = mx.contiguous(mx.concatenate([g.weight, u.weight], axis=0))
    shs = mx.contiguous(mx.concatenate([g.scales, u.scales], axis=0))
    shb = mx.contiguous(mx.concatenate([g.biases, u.biases], axis=0))

    gate, seg = mlp.gate, mlp.shared_expert_gate
    pad = (-(mlp.num_experts + 1)) % 8
    parts_w = [gate.weight, seg.weight]
    parts_s = [gate.scales, seg.scales]
    parts_b = [gate.biases, seg.biases]
    if pad:
        parts_w.append(mx.zeros((pad,) + gate.weight.shape[1:],
                                dtype=gate.weight.dtype))
        parts_s.append(mx.zeros((pad,) + gate.scales.shape[1:],
                                dtype=gate.scales.dtype))
        parts_b.append(mx.zeros((pad,) + gate.biases.shape[1:],
                                dtype=gate.biases.dtype))
    rtw = mx.contiguous(mx.concatenate(parts_w, axis=0))
    rts = mx.contiguous(mx.concatenate(parts_s, axis=0))
    rtb = mx.contiguous(mx.concatenate(parts_b, axis=0))
    mx.eval(shw, shs, shb, rtw, rts, rtb)

    z = mx.zeros((8,), dtype=mx.uint32)
    mlp._sh_w, mlp._sh_s, mlp._sh_b = shw, shs, shb
    mlp._rt_w, mlp._rt_s, mlp._rt_b = rtw, rts, rtb
    mlp._sh_gs, mlp._sh_bits = GS, SH_BITS
    mlp._rt_gs, mlp._rt_bits = GS, RT_BITS
    for m in (g, u, gate, seg):
        m.weight = z
        m.scales = z
        m.biases = z
    mx.clear_cache()


def _patch_block() -> None:
    """Stock-graph SparseMoeBlock forward over the fused weights (any S):
    the same ops in the same order as stock, so prefill parity is at plain
    matmul-tiling rounding level. Used for S>1 and as the safety net when
    the decode kernels are unavailable."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import qwen3_next as q3n

    if getattr(q3n.Qwen3NextSparseMoeBlock.__call__, "_chad_moe_fused", False):
        return  # already patched; re-wrapping would stack guards
    stock_call = q3n.Qwen3NextSparseMoeBlock.__call__

    def call(self, x):
        if not hasattr(self, "_rt_w"):
            return stock_call(self, x)
        E = self.num_experts
        rg = mx.quantized_matmul(x, self._rt_w, scales=self._rt_s,
                                 biases=self._rt_b, transpose=True,
                                 group_size=self._rt_gs, bits=self._rt_bits)
        gates = mx.softmax(rg[..., :E], axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        gu = mx.quantized_matmul(x, self._sh_w, scales=self._sh_s,
                                 biases=self._sh_b, transpose=True,
                                 group_size=self._sh_gs, bits=self._sh_bits)
        g, u = mx.split(gu, 2, axis=-1)
        sh = self.shared_expert.down_proj(nn.silu(g) * u)
        return y + mx.sigmoid(rg[..., E:E + 1]) * sh

    call._chad_moe_fused = True  # type: ignore[attr-defined]
    q3n.Qwen3NextSparseMoeBlock.__call__ = call  # type: ignore[method-assign]


def _run_a(ka, dt, x_flat, ex, sh, iu):
    return ka(inputs=[x_flat, *ex, *sh, iu], template=[("T", dt)],
              grid=_A_GRID, threadgroup=_TG,
              output_shapes=[(TOP_K + 1, INTER)], output_dtypes=[dt])[0]


def _run_b(kb, dt, h_flat, dn, sdn, iu, sc, seg, resid):
    return kb(inputs=[h_flat, *dn, *sdn, iu, sc, seg, resid],
              template=[("T", dt)], grid=_B_GRID, threadgroup=_TG,
              output_shapes=[(HID,)], output_dtypes=[dt])[0]


def _build_kernels(model) -> Optional[tuple]:
    """JIT-build both kernels and force Metal compilation with a dummy call;
    any failure returns None (callers fall back to the stock-graph body)."""
    import mlx.core as mx
    try:
        ka = mx.fast.metal_kernel(
            name="chad_moe_gu", header=_HEADER,
            input_names=["x", "wq2", "s2", "b2", "wq6", "s6", "b6", "inds"],
            output_names=["h"], source=_K_A)
        kb = mx.fast.metal_kernel(
            name="chad_moe_dn", header=_HEADER,
            input_names=["h", "wq2", "s2", "b2", "wq6", "s6", "b6",
                         "inds", "sc", "seg", "resid"],
            output_names=["y"], source=_K_B)
        mlp = model.language_model.model.layers[0].mlp
        dt = mlp._rt_s.dtype
        x0 = mx.zeros((HID,), dtype=dt)
        i0 = mx.arange(TOP_K, dtype=mx.uint32)
        sw = mlp.switch_mlp
        h0 = _run_a(ka, dt, x0,
                    (sw._fused_w, sw._fused_s, sw._fused_b),
                    (mlp._sh_w, mlp._sh_s, mlp._sh_b), i0)
        dp = sw.down_proj
        sd = mlp.shared_expert.down_proj
        y0 = _run_b(kb, dt, h0.reshape(-1),
                    (dp.weight, dp.scales, dp.biases),
                    (sd.weight, sd.scales, sd.biases), i0,
                    mx.zeros((TOP_K,), dtype=mx.float32),
                    mx.zeros((1,), dtype=mx.float32), x0)
        mx.eval(y0)
        return ka, kb
    except Exception as e:  # noqa: BLE001
        log.warning("moe-fused kernel build failed (%s); stock-graph body", e)
        return None


def _build_step(layer, kernels):
    """Compiled S==1/B==1 MoE block step: rms_norm -> router|seg qmv ->
    stock softmax over all E logits -> top-8 + renormalize -> kernel A ->
    kernel B."""
    import mlx.core as mx

    ka, kb = kernels
    mlp = layer.mlp
    sw = mlp.switch_mlp
    dp = sw.down_proj
    sd = mlp.shared_expert.down_proj
    ln_w = layer.post_attention_layernorm.weight
    ln_eps = layer.post_attention_layernorm.eps
    E = mlp.num_experts
    rt = (mlp._rt_w, mlp._rt_s, mlp._rt_b)
    sh = (mlp._sh_w, mlp._sh_s, mlp._sh_b)
    ex = (sw._fused_w, sw._fused_s, sw._fused_b)
    dn = (dp.weight, dp.scales, dp.biases)
    sdn = (sd.weight, sd.scales, sd.biases)
    dt = mlp._rt_s.dtype

    rt_gs, rt_bits = mlp._rt_gs, mlp._rt_bits

    def fwd(hin):
        x = mx.fast.rms_norm(hin, ln_w, ln_eps)
        rg = mx.quantized_matmul(x, rt[0], scales=rt[1], biases=rt[2],
                                 transpose=True, group_size=rt_gs,
                                 bits=rt_bits)
        # routing identical to stock, bit for bit — scores enter the kernel
        # as an input, so keeping stock's exact rounding here costs nothing
        gates = mx.softmax(rg[..., :E], axis=-1, precise=True)
        inds = mx.argpartition(gates, kth=-TOP_K, axis=-1)[..., -TOP_K:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = (scores / scores.sum(axis=-1, keepdims=True)).astype(mx.float32)
        iu = inds.reshape(-1).astype(mx.uint32)
        hmid = _run_a(ka, dt, x.reshape(-1), ex, sh, iu)
        y = _run_b(kb, dt, hmid.reshape(-1), dn, sdn, iu,
                   scores.reshape(-1),
                   rg[..., E].astype(mx.float32).reshape(-1),
                   hin.reshape(-1))
        return y.reshape(1, 1, HID)

    return mx.compile(fwd)
