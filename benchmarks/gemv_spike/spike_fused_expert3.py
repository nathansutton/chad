"""GEMV-fusion spike, part 7: A+/B+ — shared expert folded into the fused pair.

Extends part 6:
  A+: one dispatch computes gate|up for the 8 routed experts (2-bit branch,
      512 tgs) AND the shared expert (6-bit branch, 64 tgs; mlx's exact 6-bit
      byte-split qdot), writing h as (9, 512) — slot 8 is the shared expert.
  B+: one dispatch computes all down-projections (2-bit routed + 6-bit shared),
      applies routing scores, sigmoid(seg) for the shared expert, and the
      residual add — the entire post-router MoE block in 2 dispatches.

Stock equivalent being replaced (inside the compiled _moe_body): 2 gather_qmm,
split/silu/mul, weighted-sum, 3 shared qmm, silu-mul, seg sigmoid-mul, final
adds — ~10 dispatches.

Validated against the stock op graph; benchmarked with the honest harness.

Run (from benchmarks/gemv_spike):
  uv run --project ../.. python spike_fused_expert3.py
"""
import mlx.core as mx
import mlx.nn as nn

from spike_qmv_tune2 import pin_clocks, timed

GS = 64
PEAK_GBS = 273.0
HID = 2048
E = 256
TOP_K = 8
INTER = 512
NGU = 2 * INTER

HEADER = """
#include <metal_simdgroup>
using namespace metal;
"""

# ---- A+ : routed gate|up (2-bit) + shared gate|up (6-bit) -> h (9, 512)
K_A = """
  constexpr int RPS = 4;
  constexpr int NSG = 2;
  constexpr int K = 2048;
  constexpr int group_size = 64;
  constexpr int blocks_per_slot = 512 / (NSG * RPS);   // 64

  const int simd_gid = simdgroup_index_in_threadgroup;
  const int simd_lid = thread_index_in_simdgroup;
  const int tgy = (int)threadgroup_position_in_grid.y;

  if (tgy < 8 * blocks_per_slot) {
    // ---------- routed experts, 2-bit, PPT=2 (vpt=32, 2 block iters)
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
    // ---------- shared expert, 6-bit, PPT=2 (vpt=8, 8 block iters)
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

# ---- B+ : routed down (2-bit) + shared down (6-bit) + scores + seg + resid
K_B = """
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
    // routed experts: 2-bit, vpt=16, single block covers K=512
    constexpr int pack_factor = 16;
    constexpr int bytes_per_pack = 4;
    constexpr int values_per_thread = 16;
    constexpr int scale_step_per_thread = group_size / values_per_thread;
    constexpr int in_vec_size_w = K * bytes_per_pack / pack_factor;

    thread float x_thread[values_per_thread];
    for (int es = 0; es < 8; es++) {
      const uint e = inds[es];
      const float sc_e = sc[es];
      const device T* hl = h + es * K + simd_lid * values_per_thread;
      float sum = 0.0f;
      for (int i = 0; i < values_per_thread; i += 4) {
        sum += hl[i] + hl[i + 1] + hl[i + 2] + hl[i + 3];
        x_thread[i] = hl[i];
        x_thread[i + 1] = hl[i + 1] / 4.0f;
        x_thread[i + 2] = hl[i + 2] / 16.0f;
        x_thread[i + 3] = hl[i + 3] / 64.0f;
      }
      const device uint8_t* ws = (const device uint8_t*)wq2 +
          ((size_t)e * 2048 + rbase) * in_vec_size_w +
          simd_lid * bytes_per_pack;
      const device T* sl = s2 + ((size_t)e * 2048 + rbase) * in_vec_size_g +
          simd_lid / scale_step_per_thread;
      const device T* bl = b2 + ((size_t)e * 2048 + rbase) * in_vec_size_g +
          simd_lid / scale_step_per_thread;
      for (int row = 0; row < RPS; row++) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        float accum = 0.0f;
        for (int i = 0; i < (values_per_thread / 4); i++) {
          accum += (x_thread[4 * i] * (wl[i] & 0x03) +
                    x_thread[4 * i + 1] * (wl[i] & 0x0c) +
                    x_thread[4 * i + 2] * (wl[i] & 0x30) +
                    x_thread[4 * i + 3] * (wl[i] & 0xc0));
        }
        result[row] += sc_e * (float(sl[row * in_vec_size_g]) * accum +
                               float(bl[row * in_vec_size_g]) * sum);
      }
    }
  }

  {
    // shared expert: 6-bit, vpt=8, 2 block iters over K=512
    constexpr int pack_factor = 4;
    constexpr int bytes_per_pack = 3;
    constexpr int values_per_thread = 8;
    constexpr int block_size = values_per_thread * 32;   // 256
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


def build():
    ka = mx.fast.metal_kernel(
        name="moe_gu3", header=HEADER,
        input_names=["x", "wq2", "s2", "b2", "wq6", "s6", "b6", "inds"],
        output_names=["h"], source=K_A)
    kb = mx.fast.metal_kernel(
        name="moe_dn3", header=HEADER,
        input_names=["h", "wq2", "s2", "b2", "wq6", "s6", "b6",
                     "inds", "sc", "seg", "resid"],
        output_names=["y"], source=K_B)
    return ka, kb


def make_weights():
    """One replica: routed 2-bit (gate|up, down) + shared 6-bit (gate|up, down)."""
    gu2 = mx.quantize(mx.random.normal((E * NGU, HID)).astype(mx.bfloat16),
                      group_size=GS, bits=2)
    dn2 = mx.quantize(mx.random.normal((E * HID, INTER)).astype(mx.bfloat16),
                      group_size=GS, bits=2)
    gu6 = mx.quantize(mx.random.normal((NGU, HID)).astype(mx.bfloat16) * 0.3,
                      group_size=GS, bits=6)
    dn6 = mx.quantize(mx.random.normal((HID, INTER)).astype(mx.bfloat16) * 0.3,
                      group_size=GS, bits=6)
    r = (tuple(a.reshape(E, NGU, -1) for a in gu2)
         + tuple(a.reshape(E, HID, -1) for a in dn2)
         + tuple(gu6) + tuple(dn6))
    mx.eval(r)
    return r


def fused(ka, kb, x, w, inds, scores, seg):
    guw2, gus2, gub2, dw2, ds2, db2 = w[:6]
    guw6, gus6, gub6 = w[6:9]
    dnw6, dns6, dnb6 = w[9:12]
    h = ka(inputs=[x.reshape(-1), guw2, gus2, gub2, guw6, gus6, gub6,
                   inds.reshape(-1)],
           template=[("T", mx.bfloat16)],
           grid=(32, 2 * (8 * 64 + 64), 1), threadgroup=(32, 2, 1),
           output_shapes=[(TOP_K + 1, INTER)], output_dtypes=[mx.bfloat16])[0]
    y = kb(inputs=[h.reshape(-1), dw2, ds2, db2, dnw6, dns6, dnb6,
                   inds.reshape(-1), scores.astype(mx.float32).reshape(-1),
                   seg.astype(mx.float32).reshape(-1), x.reshape(-1)],
           template=[("T", mx.bfloat16)],
           grid=(32, 2 * (2048 // (2 * 8)), 1), threadgroup=(32, 2, 1),
           output_shapes=[(HID,)], output_dtypes=[mx.bfloat16])[0]
    return y.reshape(1, 1, HID)


def stock(x, w, inds, scores, seg):
    """Post-router stock subgraph: experts + shared expert + residual,
    mirroring _moe_body's ops."""
    guw2, gus2, gub2, dw2, ds2, db2 = w[:6]
    guw6, gus6, gub6 = w[6:9]
    dnw6, dns6, dnb6 = w[9:12]
    sh_g = (guw6[:INTER], gus6[:INTER], gub6[:INTER])
    sh_u = (guw6[INTER:], gus6[INTER:], gub6[INTER:])

    xe = mx.expand_dims(x, (-2, -3))
    gu = mx.gather_qmm(xe, guw2, gus2, gub2, rhs_indices=inds, transpose=True,
                       group_size=GS, bits=2, sorted_indices=False)
    xg, xu = mx.split(gu, 2, axis=-1)
    hh = nn.silu(xg) * xu
    yy = mx.gather_qmm(hh, dw2, ds2, db2, rhs_indices=inds, transpose=True,
                       group_size=GS, bits=2, sorted_indices=False)
    y = (yy.squeeze(-2) * scores[..., None]).sum(axis=-2)

    def qmm6(v, t):
        return mx.quantized_matmul(v, t[0], scales=t[1], biases=t[2],
                                   transpose=True, group_size=GS, bits=6)
    sh = qmm6(nn.silu(qmm6(x, sh_g)) * qmm6(x, sh_u), (dnw6, dns6, dnb6))
    return x + y + mx.sigmoid(seg) * sh


def main():
    mx.random.seed(9)
    x = mx.random.normal((1, 1, HID)).astype(mx.bfloat16)
    inds = mx.array([[[3, 17, 42, 99, 120, 180, 201, 250]]], dtype=mx.uint32)
    scores = mx.softmax(mx.random.normal((1, 1, TOP_K)), axis=-1).astype(mx.bfloat16)
    seg = mx.array([[[0.3]]], dtype=mx.bfloat16)

    per_mb = (E * (NGU * HID + HID * INTER) * (2 / 8 + 4 / GS)) / 1e6
    nrep = max(3, int(384 / per_mb) + 1)
    reps = [make_weights() for _ in range(nrep)]
    mx.clear_cache()

    ka, kb = build()
    scl = mx.compile(lambda w: stock(x, w, inds, scores, seg))

    ref = scl(reps[0])
    out = fused(ka, kb, x, reps[0], inds, scores, seg)
    mx.eval(ref, out)
    r32 = ref.reshape(-1).astype(mx.float32)
    err = mx.abs(out.reshape(-1).astype(mx.float32) - r32).max().item()
    scale = mx.abs(r32).max().item()
    print(f"numerics: max abs {err:.4f} on |y|max {scale:.1f} "
          f"({err/scale:.2e} rel; bf16 class)")

    nbytes = (TOP_K * (NGU * HID + HID * INTER) * (2 / 8 + 4 / GS)
              + (NGU * HID + HID * INTER) * (6 / 8 + 4 / GS))
    print(f"pinning clocks... ({nrep} replicas)")
    pin_clocks()
    res = timed({"stock post-router": lambda i: scl(reps[i % nrep]),
                 "fused A+ B+": lambda i: fused(ka, kb, x, reps[i % nrep],
                                                inds, scores, seg)},
                rounds=10, inner=12)
    for name, t in res.items():
        print(f"{name:18s} {t*1e6:7.1f} us  {nbytes/t/1e9:6.1f} GB/s "
              f"({100*nbytes/t/1e9/PEAK_GBS:4.1f}%)")
    ts, tf = res["stock post-router"], res["fused A+ B+"]
    print(f"\nspeedup {ts/tf:.2f}x -> x40 layers: "
          f"{(ts-tf)*40*1e3:.2f} ms/token saved")


if __name__ == "__main__":
    main()
