"""GEMV-fusion spike, part 4: byte-LUT dequant for 2-bit qmv.

Honest measurement (part 3b) shows mlx's 2-bit qmv is instruction-bound at
~437 Gelem/s (50% of DRAM peak) and its schedule is already locally optimal.
The remaining lever is fewer instructions per element: a 256-entry float4
lookup table maps one weight byte -> 4 dequantized values, replacing
4x(AND+FMA)+x-prescale (~11 uops/4 elems) with load+lookup+dot
(~6 uops/4 elems).

Variants: LUT in threadgroup memory (bank-conflict risk on random bytes) and
x staged as float4 registers. Same affine math as stock -> validated against
mx.quantized_matmul.

Run (from benchmarks/gemv_spike):  uv run --project ../.. python spike_qmv_lut.py
"""
import time

import mlx.core as mx

from spike_qmv_tune2 import make_replicas, pin_clocks, timed

GS = 64
PEAK_GBS = 273.0
N, K = 8192, 2048

HEADER = """
#include <metal_simdgroup>
using namespace metal;
"""

LUT_BODY = """
  constexpr int PPT = {PPT};           // uint32 packs per thread per block
  constexpr int RPS = {RPS};
  constexpr int NSG = {NSG};
  constexpr int K = {K};
  constexpr int group_size = 64;
  constexpr int pack_factor = 16;
  constexpr int bytes_per_pack = 4;
  constexpr int values_per_thread = pack_factor * PPT;
  constexpr int vecs_per_thread = values_per_thread / 4;
  constexpr int block_size = values_per_thread * 32;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const int simd_gid = simdgroup_index_in_threadgroup;
  const int simd_lid = thread_index_in_simdgroup;

  threadgroup float4 lut[256];
  {{
    const int t = simd_gid * 32 + simd_lid;
    for (int i = t; i < 256; i += 32 * NSG) {{
      lut[i] = float4(float(i & 3), float((i >> 2) & 3),
                      float((i >> 4) & 3), float(i >> 6));
    }}
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device uint8_t* ws = (const device uint8_t*)wq;
  thread float4 xt[vecs_per_thread];
  thread float result[RPS] = {{0}};

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

  for (int k = 0; k < K; k += block_size) {{
    float sum = 0.0f;
    for (int i = 0; i < vecs_per_thread; i++) {{
      float4 v = float4(xl[4 * i], xl[4 * i + 1], xl[4 * i + 2], xl[4 * i + 3]);
      sum += v.x + v.y + v.z + v.w;
      xt[i] = v;
    }}
    for (int row = 0; row < RPS; row++) {{
      const device uint8_t* wl = wl0 + row * in_vec_size_w;
      float accum = 0.0f;
      for (int i = 0; i < vecs_per_thread; i++) {{
        accum += dot(lut[wl[i]], xt[i]);
      }}
      result[row] += float(sl0[row * in_vec_size_g]) * accum +
          float(bl0[row * in_vec_size_g]) * sum;
    }}
    wl0 += block_size * bytes_per_pack / pack_factor;
    sl0 += block_size / group_size;
    bl0 += block_size / group_size;
    xl += block_size;
  }}

  for (int row = 0; row < RPS; row++) {{
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {{
      y[out_row + row] = static_cast<T>(result[row]);
    }}
  }}
"""


def main():
    mx.random.seed(5)
    x = mx.random.normal((1, 1, K)).astype(mx.bfloat16)
    xf = x.reshape(-1)
    nb2 = N * K * 2 / 8 + 2 * (N * K // GS) * 2

    print("building replicas...")
    reps = make_replicas(2)
    nrep = len(reps)

    def stock(i):
        wq, s, b = reps[i % nrep]
        return mx.quantized_matmul(x, wq, scales=s, biases=b, transpose=True,
                                   group_size=GS, bits=2)

    cands = {"stock@2": stock}
    for PPT, RPS, NSG in [(1, 4, 2), (1, 4, 4), (1, 8, 2),
                          (2, 4, 2), (2, 8, 2), (2, 4, 4)]:
        body = LUT_BODY.format(PPT=PPT, RPS=RPS, NSG=NSG, K=K)
        kern = mx.fast.metal_kernel(
            name=f"qmv2lut_p{PPT}r{RPS}s{NSG}", header=HEADER,
            input_names=["x", "wq", "scales", "biases"],
            output_names=["y"], source=body)

        def run(i, kern=kern, NSG=NSG, RPS=RPS):
            wq, s, b = reps[i % nrep]
            return kern(inputs=[xf, wq, s, b], template=[("T", mx.bfloat16)],
                        grid=(32, NSG * (N // (NSG * RPS)), 1),
                        threadgroup=(32, NSG, 1),
                        output_shapes=[(N,)], output_dtypes=[mx.bfloat16])[0]
        ref, out = stock(0), run(0)
        mx.eval(ref, out)
        err = mx.abs(out.astype(mx.float32)
                     - ref.reshape(-1).astype(mx.float32)).max().item()
        if err > 4.0:
            print(f"lut p{PPT} rps{RPS} nsg{NSG}: numerics ERR {err:.4f}, skip")
            continue
        cands[f"lut p{PPT} rps{RPS} nsg{NSG}"] = run

    print("pinning clocks...")
    pin_clocks()
    res = timed(cands)
    print(f"\n{'variant':22s} {'us':>7s} {'GB/s':>7s} {'%peak':>6s} {'Gelem/s':>8s}")
    for name, t in res.items():
        print(f"{name:22s} {t*1e6:7.1f} {nb2/t/1e9:7.1f} "
              f"{100*nb2/t/1e9/PEAK_GBS:5.1f}% {N*K/t/1e9:8.1f}")


if __name__ == "__main__":
    main()
