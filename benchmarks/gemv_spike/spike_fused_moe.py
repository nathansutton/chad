"""GEMV-fusion spike, part 2: fused MoE decode kernel prototype.

Replaces the S=1 expert cluster — gather_qmm(gate|up) + split + silu*mul +
gather_qmm(down) + weighted-sum — with two custom Metal dispatches:

  k1: for each of the top-8 experts, the 2-bit gate|up GEMV (1024 rows x 2048)
      with x staged once in threadgroup memory, silu(g)*u epilogue fused.
      All 8 experts in one dispatch (8 experts x 8 row-blocks = 64 tgs), so the
      GPU sees one 8192-row GEMV instead of 8 tiny 1024-row ones.
  k2: the 2-bit down GEMV (2048 rows x 512) for all 8 experts, each output
      block loops over experts and applies routing scores inline — the weighted
      combine costs zero extra traffic and no atomics.

Validates numerics against the stock mx op-graph, then benchmarks both.

Run:  uv run python benchmarks/gemv_spike/spike_fused_moe.py
"""
import time

import mlx.core as mx
import mlx.nn as nn

GS = 64
HID = 2048
E = 256
TOP_K = 8
INTER = 512          # per-expert intermediate
NGU = 2 * INTER      # fused gate|up rows
PEAK_GBS = 273.0

HEADER = """
#include <metal_simdgroup>
#include <metal_math>
using namespace metal;
"""

# k1: threadgroup (32, 8, 1); grid tgs (x=NGU/2/BJ blocks, y=8 expert slots).
# Each simdgroup handles 8 j's; each j needs row g=j and row u=j+512.
# Lane l owns quant group l of K=2048 (4 uint32 = 64 values), so scale/bias
# lookups are one per lane per row.
K1_SRC = """
  constexpr int KD = 2048;          // input dim
  constexpr int BJ = 64;            // j's per threadgroup
  constexpr int PW = KD / 16;       // 128 packed u32 per row
  constexpr int GW = KD / 64;       // 32 groups per row

  const int eslot = threadgroup_position_in_grid.y;
  const int jbase = threadgroup_position_in_grid.x * BJ;
  const int sg = simdgroup_index_in_threadgroup;   // 0..7
  const int lane = thread_index_in_simdgroup;       // 0..31
  const uint e = inds[eslot];

  threadgroup float xs[KD];
  threadgroup float xgsum[GW];
  // stage x (bf16 -> f32), 256 threads x 8 elements
  const int t = sg * 32 + lane;
  for (int i = 0; i < 8; i++) xs[t * 8 + i] = float(x[t * 8 + i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < GW) {
    float acc = 0.0f;
    for (int i = 0; i < 64; i++) acc += xs[t * 64 + i];
    xgsum[t] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device uint* wbase = wq + (size_t)e * (NGU_C * PW);
  const device T* sbase = scales + (size_t)e * (NGU_C * GW);
  const device T* bbase = biases + (size_t)e * (NGU_C * GW);

  for (int r = 0; r < 8; r++) {
    const int j = jbase + sg * 8 + r;
    float rv[2];
    for (int hf = 0; hf < 2; hf++) {
      const int row = j + hf * 512;
      const device uint* wr = wbase + (size_t)row * PW + lane * 4;
      float qdot = 0.0f;
      for (int p = 0; p < 4; p++) {
        uint w = wr[p];
        const int off = lane * 64 + p * 16;
        for (int i = 0; i < 16; i++) {
          qdot += float((w >> (2 * i)) & 3u) * xs[off + i];
        }
      }
      const float s = float(sbase[(size_t)row * GW + lane]);
      const float b = float(bbase[(size_t)row * GW + lane]);
      rv[hf] = simd_sum(s * qdot + b * xgsum[lane]);
    }
    if (lane == 0) {
      const float g = rv[0];
      const float u = rv[1];
      h[eslot * 512 + j] = (g / (1.0f + metal::exp(-g))) * u;
    }
  }
"""

# k2: threadgroup (32, 8, 1); grid tgs (x=2048/BN row blocks).
# K=512: lane l owns u32 l (16 values); quant group = l/4. h staged for all 8
# experts (16 KB); per-group h sums precomputed for the affine bias term.
K2_SRC = """
  constexpr int KD = 512;
  constexpr int BN = 64;            // output rows per threadgroup
  constexpr int PW = KD / 16;       // 32 packed u32 per row
  constexpr int GW = KD / 64;       // 8 groups per row

  const int nbase = threadgroup_position_in_grid.x * BN;
  const int sg = simdgroup_index_in_threadgroup;
  const int lane = thread_index_in_simdgroup;
  const int t = sg * 32 + lane;

  threadgroup float hs[8 * KD];
  threadgroup float hgsum[8 * GW];
  for (int i = 0; i < 16; i++) hs[t * 16 + i] = h[t * 16 + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 8 * GW) {
    float acc = 0.0f;
    for (int i = 0; i < 64; i++) acc += hs[t * 64 + i];
    hgsum[t] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int grp = lane / 4;
  float acc = 0.0f;
  for (int r = 0; r < 8; r++) {
    const int row = nbase + sg * 8 + r;
    float rowacc = 0.0f;
    for (int es = 0; es < 8; es++) {
      const uint e = inds[es];
      const device uint* wr = wq + ((size_t)e * 2048 + row) * PW + lane;
      const uint w = *wr;
      const threadgroup float* hb = hs + es * KD + lane * 16;
      float qdot = 0.0f;
      for (int i = 0; i < 16; i++) {
        qdot += float((w >> (2 * i)) & 3u) * hb[i];
      }
      const float s = float(scales[((size_t)e * 2048 + row) * GW + grp]);
      const float b = float(biases[((size_t)e * 2048 + row) * GW + grp]);
      // bias term: each of the 4 lanes in a group must contribute b*hgsum/4
      rowacc += float(sc[es]) * (s * qdot + 0.25f * b * hgsum[es * GW + grp]);
    }
    const float rv = simd_sum(rowacc);
    if (lane == 0) y[row] = static_cast<T>(rv);
  }
"""


def build_kernels():
    k1 = mx.fast.metal_kernel(
        name="moe_gu_fused", header=HEADER,
        input_names=["x", "wq", "scales", "biases", "inds"],
        output_names=["h"], source=K1_SRC.replace("NGU_C", str(NGU)))
    k2 = mx.fast.metal_kernel(
        name="moe_down_fused", header=HEADER,
        input_names=["h", "wq", "scales", "biases", "inds", "sc"],
        output_names=["y"], source=K2_SRC)
    return k1, k2


def fused_moe(k1, k2, x, guw, gus, gub, dw, ds, db, inds, scores, dtype):
    h = k1(inputs=[x.reshape(-1), guw, gus, gub, inds.reshape(-1)],
           template=[("T", dtype)],
           grid=(8 * 32, 8 * 8, 1), threadgroup=(32, 8, 1),
           output_shapes=[(TOP_K, INTER)], output_dtypes=[mx.float32])[0]
    y = k2(inputs=[h.reshape(-1), dw, ds, db, inds.reshape(-1),
                   scores.astype(mx.float32).reshape(-1)],
           template=[("T", dtype)],
           grid=(32 * 32, 8, 1), threadgroup=(32, 8, 1),
           output_shapes=[(HID,)], output_dtypes=[dtype])[0]
    return y.reshape(1, 1, HID)


def stock_moe(x, guw, gus, gub, dw, ds, db, inds, scores):
    xe = mx.expand_dims(x, (-2, -3))
    gu = mx.gather_qmm(xe, guw, gus, gub, rhs_indices=inds, transpose=True,
                       group_size=GS, bits=2, sorted_indices=False)
    xg, xu = mx.split(gu, 2, axis=-1)
    hh = nn.silu(xg) * xu
    yy = mx.gather_qmm(hh, dw, ds, db, rhs_indices=inds, transpose=True,
                       group_size=GS, bits=2, sorted_indices=False)
    return (yy.squeeze(-2) * scores[..., None]).sum(axis=-2)


def bench(fn, iters=40, warmup=8, K=24):
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval([fn() for _ in range(K)])
        ts.append((time.perf_counter() - t0) / K)
    ts.sort()
    return ts[len(ts) // 2]


def bench_serial(fn, x, iters=30, warmup=5, K=24):
    def chained():
        cur = x
        outs = []
        for _ in range(K):
            y = fn(cur)
            outs.append(y)
            cur = (x + 0.0 * mx.sum(y)).astype(x.dtype)
        return outs
    for _ in range(warmup):
        mx.eval(chained())
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(chained())
        ts.append((time.perf_counter() - t0) / K)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    mx.random.seed(1)
    dtype = mx.bfloat16
    x = mx.random.normal((1, 1, HID)).astype(dtype)

    gu = mx.random.normal((E, NGU, HID)).astype(dtype) * 0.5
    guw, gus, gub = mx.quantize(gu.reshape(E * NGU, HID), group_size=GS, bits=2)
    guw = guw.reshape(E, NGU, -1); gus = gus.reshape(E, NGU, -1)
    gub = gub.reshape(E, NGU, -1)
    dn = mx.random.normal((E, HID, INTER)).astype(dtype) * 0.5
    dw, ds, db = mx.quantize(dn.reshape(E * HID, INTER), group_size=GS, bits=2)
    dw = dw.reshape(E, HID, -1); ds = ds.reshape(E, HID, -1)
    db = db.reshape(E, HID, -1)
    mx.eval(guw, gus, gub, dw, ds, db)
    del gu, dn
    mx.clear_cache()

    inds = mx.array([[[3, 17, 42, 99, 120, 180, 201, 250]]], dtype=mx.uint32)
    scores = mx.softmax(mx.random.normal((1, 1, TOP_K)), axis=-1).astype(dtype)

    k1, k2 = build_kernels()
    args = (guw, gus, gub, dw, ds, db, inds, scores)

    ref = stock_moe(x, *args)
    out = fused_moe(k1, k2, x, *args, dtype)
    mx.eval(ref, out)
    ref32 = ref.astype(mx.float32)
    err = mx.abs(out.astype(mx.float32) - ref32)
    rel = (err.max() / (mx.abs(ref32).max() + 1e-9)).item()
    print(f"numerics: max abs err {err.max().item():.5f}, "
          f"rel {rel:.5f}  (bf16 rounding class ~3e-3)")

    nbytes = TOP_K * (NGU * HID + HID * INTER) * (2 / 8 + 2 * 2 / GS)
    stock_c = mx.compile(lambda a: stock_moe(a, *args))
    t_stock = bench(lambda: stock_c(x))
    t_fused = bench(lambda: fused_moe(k1, k2, x, *args, dtype))
    ts_stock = bench_serial(lambda a: stock_c(a), x)
    ts_fused = bench_serial(lambda a: fused_moe(k1, k2, a, *args, dtype), x)

    print(f"\n{'':14s} {'us(tp)':>8s} {'us(ser)':>8s} {'GB/s(tp)':>9s} {'%peak':>6s}")
    for nm, tp, ser in [("stock", t_stock, ts_stock),
                        ("fused", t_fused, ts_fused)]:
        print(f"{nm:14s} {tp*1e6:8.1f} {ser*1e6:8.1f} "
              f"{nbytes/tp/1e9:9.1f} {100*nbytes/tp/1e9/PEAK_GBS:5.1f}%")
    print(f"\nspeedup: tp {t_stock/t_fused:.2f}x, serialized {ts_stock/ts_fused:.2f}x")
    print(f"per-token (x40 layers, serialized): "
          f"{(ts_stock-ts_fused)*40*1e3:.2f} ms saved")


if __name__ == "__main__":
    main()
