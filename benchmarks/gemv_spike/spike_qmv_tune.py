"""GEMV-fusion spike, part 3: retune mlx's own 2-bit qmv schedule.

mlx's qmv runs ~300G weight-elements/s at every bit width on the M4 Pro.
At 6-bit that coincides with saturated bandwidth (87% peak); at 2-bit it
leaves the bus 62% idle. The inner-loop math (load_vector/qdot) is already
minimal, so this ports mlx's exact 2-bit math into mx.fast.metal_kernel with
the SCHEDULE made tunable:

  PPT  packs (uint32) per thread per block   (mlx ships 1 for 2-bit)
  RPS  result rows per simdgroup             (mlx ships 4)
  NSG  simdgroups per threadgroup            (mlx ships 2)

and sweeps the grid, validating each variant bit-exactly against
mx.quantized_matmul (same math, same order -> must match to float rounding).

Run:  uv run python benchmarks/gemv_spike/spike_qmv_tune.py
"""
import time

import mlx.core as mx

GS = 64
PEAK_GBS = 273.0

# mlx's load_vector + qdot for bits=2, verbatim math (quantized.h), inlined so
# the spike does not depend on wheel include paths.
HEADER = """
#include <metal_simdgroup>
using namespace metal;
"""

BODY_TMPL = """
  constexpr int PPT = {PPT};
  constexpr int RPS = {RPS};
  constexpr int NSG = {NSG};
  constexpr int K = {K};
  constexpr int N = {N};
  constexpr int group_size = 64;
  constexpr int pack_factor = 16;          // 2-bit values per uint32
  constexpr int bytes_per_pack = 4;
  constexpr int values_per_thread = pack_factor * PPT;
  constexpr int block_size = values_per_thread * 32;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)wq;

  thread float x_thread[values_per_thread];
  thread float result[RPS] = {{0}};

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

  for (int k = 0; k < K; k += block_size) {{
    float sum = 0.0f;
    for (int i = 0; i < values_per_thread; i += 4) {{
      sum += xl[i] + xl[i + 1] + xl[i + 2] + xl[i + 3];
      x_thread[i] = xl[i];
      x_thread[i + 1] = xl[i + 1] / 4.0f;
      x_thread[i + 2] = xl[i + 2] / 16.0f;
      x_thread[i + 3] = xl[i + 3] / 64.0f;
    }}
    for (int row = 0; row < RPS; row++) {{
      const device uint8_t* wl = wl0 + row * in_vec_size_w;
      const device T* sl = sl0 + row * in_vec_size_g;
      const device T* bl = bl0 + row * in_vec_size_g;
      float s = sl[0];
      float b = bl[0];
      float accum = 0.0f;
      for (int i = 0; i < (values_per_thread / 4); i++) {{
        accum +=
            (x_thread[4 * i] * (wl[i] & 0x03) +
             x_thread[4 * i + 1] * (wl[i] & 0x0c) +
             x_thread[4 * i + 2] * (wl[i] & 0x30) +
             x_thread[4 * i + 3] * (wl[i] & 0xc0));
      }}
      result[row] += s * accum + b * sum;
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


def bench(fn, iters=30, warm=6, K=24):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval([fn() for _ in range(K)])
        ts.append((time.perf_counter() - t0) / K)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    mx.random.seed(3)
    N, K = 8192, 2048
    w = mx.random.normal((N, K)).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=GS, bits=2)
    mx.eval(wq, s, b)
    del w
    x = mx.random.normal((1, 1, K)).astype(mx.bfloat16)
    nbytes = N * K * 2 / 8 + 2 * (N * K // GS) * 2

    ref = mx.quantized_matmul(x, wq, scales=s, biases=b, transpose=True,
                              group_size=GS, bits=2)
    t_ref = bench(lambda: mx.quantized_matmul(
        x, wq, scales=s, biases=b, transpose=True, group_size=GS, bits=2))
    print(f"{'variant':22s} {'us':>7s} {'GB/s':>7s} {'%peak':>6s} {'Gelem/s':>8s}")
    print(f"{'mlx qmv (stock)':22s} {t_ref*1e6:7.1f} {nbytes/t_ref/1e9:7.1f} "
          f"{100*nbytes/t_ref/1e9/PEAK_GBS:5.1f}% {N*K/t_ref/1e9:8.1f}")

    xf = x.reshape(-1)
    for PPT in (1, 2, 4):
        for RPS in (4, 8, 16):
            for NSG in (2, 4, 8):
                if N % (NSG * RPS):
                    continue
                if K % (16 * PPT * 32):
                    continue
                body = BODY_TMPL.format(PPT=PPT, RPS=RPS, NSG=NSG, K=K, N=N)
                kern = mx.fast.metal_kernel(
                    name=f"qmv2_p{PPT}r{RPS}s{NSG}", header=HEADER,
                    input_names=["x", "wq", "scales", "biases"],
                    output_names=["y"], source=body)

                def run(kern=kern, NSG=NSG, RPS=RPS):
                    return kern(inputs=[xf, wq, s, b],
                                template=[("T", mx.bfloat16)],
                                grid=(32, NSG * (N // (NSG * RPS)), 1),
                                threadgroup=(32, NSG, 1),
                                output_shapes=[(N,)],
                                output_dtypes=[mx.bfloat16])[0]
                try:
                    out = run()
                    mx.eval(out)
                except Exception as e:
                    print(f"p{PPT} r{RPS} s{NSG}: FAILED {e}")
                    continue
                err = mx.abs(out.astype(mx.float32)
                             - ref.reshape(-1).astype(mx.float32)).max().item()
                t = bench(run)
                tag = f"p{PPT} rps{RPS} nsg{NSG}"
                flag = "" if err < 0.05 else f"  ERR={err:.3f}"
                print(f"{tag:22s} {t*1e6:7.1f} {nbytes/t/1e9:7.1f} "
                      f"{100*nbytes/t/1e9/PEAK_GBS:5.1f}% {N*K/t/1e9:8.1f}{flag}")

    # bit-width elem-rate table for reference (stock mlx)
    print("\nstock mlx qmv elem rates at (8192x2048):")
    for bits in (2, 3, 4, 5, 6, 8):
        w = mx.random.normal((N, K)).astype(mx.bfloat16)
        q = mx.quantize(w, group_size=GS, bits=bits)
        mx.eval(q)
        del w
        t = bench(lambda q=q, bits=bits: mx.quantized_matmul(
            x, q[0], scales=q[1], biases=q[2], transpose=True,
            group_size=GS, bits=bits))
        nb = N * K * bits / 8 + 2 * (N * K // GS) * 2
        print(f"  {bits}-bit: {t*1e6:7.1f} us  {nb/t/1e9:6.1f} GB/s "
              f"({100*nb/t/1e9/PEAK_GBS:4.1f}%)  {N*K/t/1e9:6.1f} Gelem/s")
        del q
        mx.clear_cache()


if __name__ == "__main__":
    main()
