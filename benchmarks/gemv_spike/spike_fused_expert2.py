"""GEMV-fusion spike, part 6: fused expert kernels built on mlx's inner loop.

Part 5 measured the real expert cluster at 71.7 GB/s (26% of peak) while plain
2-bit qmv plateaus at 137 GB/s. This rebuilds the part-2 fusion idea, but with
mlx's exact qmv_fast inner loop (prescaled x_thread, byte AND+FMA qdot, per-
thread group alignment) instead of a naive loop:

  A: gate|up gathered GEMV. One dispatch for all 8 experts (512 tgs), each
     threadgroup computes RPS j's = paired rows (j, j+512) of the fused
     gate|up matrix, silu(g)*u in-register, writes h (8,512) bf16.
  B: down gathered GEMV + weighted combine. Each threadgroup owns RPS output
     rows and loops the 8 experts, scores applied in-register; y written once.

Numerics: same math and accumulation order as stock per expert -> validated
against the stock cluster at bf16 rounding tolerance.

Run (from benchmarks/gemv_spike):
  uv run --project ../.. python spike_fused_expert2.py
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

# A: threadgroup (32, NSG). tg y -> (expert slot, j block). PPT=2 -> vpt=32,
# block_size=1024, K=2048 -> 2 block iters. Each tg: RPS j's, 2*RPS row-dots.
K_A = """
  constexpr int PPT = 2;
  constexpr int RPS = 4;
  constexpr int NSG = 2;
  constexpr int K = 2048;
  constexpr int group_size = 64;
  constexpr int pack_factor = 16;
  constexpr int bytes_per_pack = 4;
  constexpr int values_per_thread = pack_factor * PPT;
  constexpr int block_size = values_per_thread * 32;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  constexpr int blocks_per_slot = 512 / (NSG * RPS);   // j blocks per expert

  const int simd_gid = simdgroup_index_in_threadgroup;
  const int simd_lid = thread_index_in_simdgroup;
  const int tgy = (int)threadgroup_position_in_grid.y;
  const int slot = tgy / blocks_per_slot;
  const int jbase = (tgy % blocks_per_slot) * (NSG * RPS) + simd_gid * RPS;
  const uint e = inds[slot];

  constexpr int in_vec_size_w = K * bytes_per_pack / pack_factor;
  constexpr int in_vec_size_g = K / group_size;
  const device uint8_t* ws = (const device uint8_t*)wq +
      (size_t)e * 1024 * in_vec_size_w;
  const device T* se = scales + (size_t)e * 1024 * in_vec_size_g;
  const device T* be = biases + (size_t)e * 1024 * in_vec_size_g;

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
  constexpr size_t uoff = (size_t)512 * in_vec_size_w;    // +512 rows
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
"""

# B: threadgroup (32, NSG). tg y -> output row block. K=512, PPT=1 -> vpt=16,
# block_size=512 -> single block iter; loops the 8 experts in-register.
K_B = """
  constexpr int RPS = 8;
  constexpr int NSG = 2;
  constexpr int K = 512;
  constexpr int group_size = 64;
  constexpr int pack_factor = 16;
  constexpr int bytes_per_pack = 4;
  constexpr int values_per_thread = 16;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const int simd_gid = simdgroup_index_in_threadgroup;
  const int simd_lid = thread_index_in_simdgroup;
  const int rbase = (int)threadgroup_position_in_grid.y * (NSG * RPS) +
      simd_gid * RPS;

  constexpr int in_vec_size_w = K * bytes_per_pack / pack_factor;
  constexpr int in_vec_size_g = K / group_size;

  thread float x_thread[values_per_thread];
  thread float result[RPS] = {0};

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
    const device uint8_t* ws = (const device uint8_t*)wq +
        ((size_t)e * 2048 + rbase) * in_vec_size_w +
        simd_lid * bytes_per_pack;
    const device T* sl = scales + ((size_t)e * 2048 + rbase) * in_vec_size_g +
        simd_lid / scale_step_per_thread;
    const device T* bl = biases + ((size_t)e * 2048 + rbase) * in_vec_size_g +
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

  for (int row = 0; row < RPS; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[rbase + row] = static_cast<T>(result[row]);
    }
  }
"""


def build():
    ka = mx.fast.metal_kernel(name="moe_gu2", header=HEADER,
                              input_names=["x", "wq", "scales", "biases",
                                           "inds"],
                              output_names=["h"], source=K_A)
    kb = mx.fast.metal_kernel(name="moe_dn2", header=HEADER,
                              input_names=["h", "wq", "scales", "biases",
                                           "inds", "sc"],
                              output_names=["y"], source=K_B)
    return ka, kb


def fused(ka, kb, x, guw, gus, gub, dw, ds, db, inds, scores):
    h = ka(inputs=[x.reshape(-1), guw, gus, gub, inds.reshape(-1)],
           template=[("T", mx.bfloat16)],
           grid=(32, 2 * 8 * (512 // (2 * 4)), 1), threadgroup=(32, 2, 1),
           output_shapes=[(TOP_K, INTER)], output_dtypes=[mx.bfloat16])[0]
    y = kb(inputs=[h.reshape(-1), dw, ds, db, inds.reshape(-1),
                   scores.astype(mx.float32).reshape(-1)],
           template=[("T", mx.bfloat16)],
           grid=(32, 2 * (2048 // (2 * 8)), 1), threadgroup=(32, 2, 1),
           output_shapes=[(HID,)], output_dtypes=[mx.bfloat16])[0]
    return y.reshape(1, 1, HID)


def main():
    mx.random.seed(7)
    x = mx.random.normal((1, 1, HID)).astype(mx.bfloat16)
    inds = mx.array([[[3, 17, 42, 99, 120, 180, 201, 250]]], dtype=mx.uint32)
    scores = mx.softmax(mx.random.normal((1, 1, TOP_K)), axis=-1).astype(mx.bfloat16)

    per_mb = (E * (NGU * HID + HID * INTER) * (2 / 8 + 4 / GS)) / 1e6
    nrep = max(3, int(384 / per_mb) + 1)
    reps = []
    for _ in range(nrep):
        gu = mx.quantize(mx.random.normal((E * NGU, HID)).astype(mx.bfloat16),
                         group_size=GS, bits=2)
        dn = mx.quantize(mx.random.normal((E * HID, INTER)).astype(mx.bfloat16),
                         group_size=GS, bits=2)
        r = tuple(a.reshape(E, NGU, -1) for a in gu) + \
            tuple(a.reshape(E, HID, -1) for a in dn)
        mx.eval(r)
        reps.append(r)
    mx.clear_cache()

    def cluster(guw, gus, gub, dw, ds, db):
        xe = mx.expand_dims(x, (-2, -3))
        gu = mx.gather_qmm(xe, guw, gus, gub, rhs_indices=inds, transpose=True,
                           group_size=GS, bits=2, sorted_indices=False)
        xg, xu = mx.split(gu, 2, axis=-1)
        hh = nn.silu(xg) * xu
        yy = mx.gather_qmm(hh, dw, ds, db, rhs_indices=inds, transpose=True,
                           group_size=GS, bits=2, sorted_indices=False)
        return (yy.squeeze(-2) * scores[..., None]).sum(axis=-2)

    ccl = mx.compile(cluster)
    ka, kb = build()

    ref = ccl(*reps[0])
    out = fused(ka, kb, x, *reps[0], inds, scores)
    mx.eval(ref, out)
    r32 = ref.reshape(-1).astype(mx.float32)
    err = mx.abs(out.reshape(-1).astype(mx.float32) - r32).max().item()
    scale = mx.abs(r32).max().item()
    print(f"numerics: max abs err {err:.5f} on |y|max {scale:.2f} "
          f"({err/scale:.2e} rel)")

    nbytes = TOP_K * (NGU * HID + HID * INTER) * (2 / 8 + 4 / GS)
    print(f"pinning clocks... ({nrep} replicas)")
    pin_clocks()
    res = timed({"stock cluster": lambda i: ccl(*reps[i % nrep]),
                 "fused A+B": lambda i: fused(ka, kb, x, *reps[i % nrep],
                                              inds, scores)},
                rounds=10, inner=12)
    for name, t in res.items():
        print(f"{name:14s} {t*1e6:7.1f} us  {nbytes/t/1e9:6.1f} GB/s "
              f"({100*nbytes/t/1e9/PEAK_GBS:4.1f}%)")
    ts, tf = res["stock cluster"], res["fused A+B"]
    print(f"\nspeedup {ts/tf:.2f}x  -> x40 layers: "
          f"{(ts-tf)*40*1e3:.2f} ms/token saved")


if __name__ == "__main__":
    main()
