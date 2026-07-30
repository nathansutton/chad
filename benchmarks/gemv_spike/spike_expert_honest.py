"""GEMV-fusion spike, part 5: honest expert-path numbers + fp16-qdot probe.

(a) Cache-busted, clock-pinned measurement of the real MoE expert cluster
    (gather_qmm gate|up + silu*mul + gather_qmm down + weighted sum, compiled)
    vs the plain-qmv 2-bit plateau (137 GB/s). The gap is what a fused
    expert kernel with mlx's own inner loop could recover.
(b) A half-math qdot variant of the part-3 kernel: if M4 dual-issues fp16,
    it shows up here; if fp16 rate == fp32 rate, it won't.

Run (from benchmarks/gemv_spike):
  uv run --project ../.. python spike_expert_honest.py
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

# part-3 kernel with the qdot arithmetic done in half (x staged as half,
# products in half, per-block accumulate promoted to float).
HALF_BODY = """
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

  const device uint8_t* ws = (const device uint8_t*)wq;
  thread half x_thread[values_per_thread];
  thread float result[RPS] = {0};

  const int simd_gid = simdgroup_index_in_threadgroup;
  const int simd_lid = thread_index_in_simdgroup;
  const int out_row = (int)threadgroup_position_in_grid.y * (NSG * RPS) +
      simd_gid * RPS;
  constexpr int in_vec_size_w = K * bytes_per_pack / pack_factor;
  constexpr int in_vec_size_g = K / group_size;

  const device uint8_t* wl0 = ws + out_row * in_vec_size_w +
      simd_lid * PPT * bytes_per_pack;
  const device T* sl0 = scales + out_row * in_vec_size_g +
      simd_lid / scale_step_per_thread;
  const device T* bl0 = biases + out_row * in_vec_size_g +
      simd_lid / scale_step_per_thread;
  const device T* xl = x + simd_lid * values_per_thread;

  for (int k = 0; k < K; k += block_size) {
    float sum = 0.0f;
    for (int i = 0; i < values_per_thread; i += 4) {
      float x0 = xl[i], x1 = xl[i + 1], x2 = xl[i + 2], x3 = xl[i + 3];
      sum += x0 + x1 + x2 + x3;
      x_thread[i] = half(x0);
      x_thread[i + 1] = half(x1 / 4.0f);
      x_thread[i + 2] = half(x2 / 16.0f);
      x_thread[i + 3] = half(x3 / 64.0f);
    }
    for (int row = 0; row < RPS; row++) {
      const device uint8_t* wl = wl0 + row * in_vec_size_w;
      half accum = 0.0h;
      for (int i = 0; i < (values_per_thread / 4); i++) {
        accum +=
            (x_thread[4 * i] * half(wl[i] & 0x03) +
             x_thread[4 * i + 1] * half(wl[i] & 0x0c) +
             x_thread[4 * i + 2] * half(wl[i] & 0x30) +
             x_thread[4 * i + 3] * half(wl[i] & 0xc0));
      }
      result[row] += float(sl0[row * in_vec_size_g]) * float(accum) +
          float(bl0[row * in_vec_size_g]) * sum;
    }
    wl0 += block_size * bytes_per_pack / pack_factor;
    sl0 += block_size / group_size;
    bl0 += block_size / group_size;
    xl += block_size;
  }

  for (int row = 0; row < RPS; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[out_row + row] = static_cast<T>(result[row]);
    }
  }
"""


def main():
    mx.random.seed(6)
    x = mx.random.normal((1, 1, HID)).astype(mx.bfloat16)

    # ---- (a) expert cluster, replicated to bust the SLC
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
    inds = mx.array([[[3, 17, 42, 99, 120, 180, 201, 250]]], dtype=mx.uint32)
    scores = mx.softmax(mx.random.normal((1, 1, TOP_K)), axis=-1).astype(mx.bfloat16)

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
    nbytes = TOP_K * (NGU * HID + HID * INTER) * (2 / 8 + 4 / GS)

    def run_cluster(i):
        return ccl(*reps[i % nrep])

    def run_gu_only(i):
        guw, gus, gub = reps[i % nrep][:3]
        xe = mx.expand_dims(x, (-2, -3))
        return mx.gather_qmm(xe, guw, gus, gub, rhs_indices=inds,
                             transpose=True, group_size=GS, bits=2,
                             sorted_indices=False)

    print(f"pinning clocks... ({nrep} replicas of {per_mb:.0f} MB)")
    pin_clocks()
    res = timed({"cluster": run_cluster, "gate|up only": run_gu_only},
                rounds=10, inner=12)
    nb_gu = TOP_K * NGU * HID * (2 / 8 + 4 / GS)
    t = res["cluster"]
    print(f"\nexpert cluster (honest): {t*1e6:7.1f} us  "
          f"{nbytes/t/1e9:6.1f} GB/s ({100*nbytes/t/1e9/PEAK_GBS:4.1f}%)  "
          f"-> x40 layers = {t*40*1e3:.2f} ms/token")
    t = res["gate|up only"]
    print(f"gate|up gather only:     {t*1e6:7.1f} us  "
          f"{nb_gu/t/1e9:6.1f} GB/s ({100*nb_gu/t/1e9/PEAK_GBS:4.1f}%)")
    del reps
    mx.clear_cache()

    # ---- (b) fp16 qdot probe at 8192x2048
    from spike_qmv_tune2 import make_replicas
    N, K = 8192, 2048
    reps2 = make_replicas(2)
    nrep2 = len(reps2)
    xk = mx.random.normal((1, 1, K)).astype(mx.bfloat16)
    xf = xk.reshape(-1)
    kern = mx.fast.metal_kernel(
        name="qmv2_half", header=HEADER,
        input_names=["x", "wq", "scales", "biases"],
        output_names=["y"], source=HALF_BODY)

    def run_half(i):
        wq, s, b = reps2[i % nrep2]
        return kern(inputs=[xf, wq, s, b], template=[("T", mx.bfloat16)],
                    grid=(32, 2 * (N // (2 * 4)), 1), threadgroup=(32, 2, 1),
                    output_shapes=[(N,)], output_dtypes=[mx.bfloat16])[0]

    def run_stock(i):
        wq, s, b = reps2[i % nrep2]
        return mx.quantized_matmul(xk, wq, scales=s, biases=b, transpose=True,
                                   group_size=GS, bits=2)
    ref, out = run_stock(0), run_half(0)
    mx.eval(ref, out)
    err = mx.abs(out.astype(mx.float32)
                 - ref.reshape(-1).astype(mx.float32)).max().item()
    pin_clocks(1.0)
    res = timed({"stock": run_stock, "half-qdot": run_half}, rounds=10)
    nb2 = N * K * 2 / 8 + 2 * (N * K // GS) * 2
    print(f"\nfp16-qdot probe (err {err:.3f} vs stock):")
    for name, t in res.items():
        print(f"  {name:12s} {t*1e6:7.1f} us  {nb2/t/1e9:6.1f} GB/s "
              f"({100*nb2/t/1e9/PEAK_GBS:4.1f}%)")


if __name__ == "__main__":
    main()
