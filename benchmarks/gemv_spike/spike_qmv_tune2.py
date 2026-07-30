"""GEMV-fusion spike, part 3b: qmv schedule sweep with honest measurement.

Part 3's numbers were contaminated two ways: repeated matvecs on one weight
matrix hit the SLC (8-bit "measured" 140% of DRAM peak), and GPU clocks ramp
mid-script (stock 2-bit "improved" 1.4x between the top and bottom of one run).
This version:
  - cycles through enough weight replicas (>= 384 MB working set) that every
    call streams from DRAM, like real decode does;
  - pins clocks with ~2 s of sustained load before measuring;
  - interleaves variant and stock timing rounds and reports medians.

Run:  uv run python benchmarks/gemv_spike/spike_qmv_tune2.py
"""
import time

import mlx.core as mx

from spike_qmv_tune import BODY_TMPL, HEADER

GS = 64
PEAK_GBS = 273.0
N, K = 8192, 2048
WSET_MB = 384


def make_replicas(bits):
    per = (N * K * bits / 8 + 2 * (N * K // GS) * 2) / 1e6
    nrep = max(4, int(WSET_MB / per) + 1)
    reps = []
    for _ in range(nrep):
        w = mx.random.normal((N, K)).astype(mx.bfloat16)
        q = mx.quantize(w, group_size=GS, bits=bits)
        mx.eval(q)
        del w
        reps.append(q)
    mx.clear_cache()
    return reps


def pin_clocks(seconds=2.0):
    a = mx.random.normal((4096, 4096))
    b = mx.random.normal((4096, 4096))
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        b = mx.matmul(a, b)
        b = b / mx.linalg.norm(b)
        mx.eval(b)


def timed(callables, rounds=12, inner=16):
    """callables: dict name -> fn(i) building one matvec on replica i.
    Interleaved rounds; returns dict name -> median sec/call."""
    acc = {k: [] for k in callables}
    idx = 0
    for _ in range(rounds):
        for name, fn in callables.items():
            outs = [fn(idx + j) for j in range(inner)]
            mx.eval(outs)  # warm graph, not timed
            t0 = time.perf_counter()
            outs = [fn(idx + inner + j) for j in range(inner)]
            mx.eval(outs)
            acc[name].append((time.perf_counter() - t0) / inner)
            idx += 2 * inner
    return {k: sorted(v)[len(v) // 2] for k, v in acc.items()}


def main():
    mx.random.seed(4)
    x = mx.random.normal((1, 1, K)).astype(mx.bfloat16)
    xf = x.reshape(-1)

    print("building replicas...")
    reps2 = make_replicas(2)
    nb2 = N * K * 2 / 8 + 2 * (N * K // GS) * 2
    nrep = len(reps2)

    def stock(i):
        wq, s, b = reps2[i % nrep]
        return mx.quantized_matmul(x, wq, scales=s, biases=b, transpose=True,
                                   group_size=GS, bits=2)

    cands = {"stock@2": stock}
    for PPT, RPS, NSG in [(1, 4, 4), (1, 8, 2), (1, 16, 4),
                          (2, 4, 2), (2, 4, 8), (2, 16, 4)]:
        body = BODY_TMPL.format(PPT=PPT, RPS=RPS, NSG=NSG, K=K, N=N)
        kern = mx.fast.metal_kernel(
            name=f"qmv2b_p{PPT}r{RPS}s{NSG}", header=HEADER,
            input_names=["x", "wq", "scales", "biases"],
            output_names=["y"], source=body)

        def run(i, kern=kern, NSG=NSG, RPS=RPS):
            wq, s, b = reps2[i % nrep]
            return kern(inputs=[xf, wq, s, b], template=[("T", mx.bfloat16)],
                        grid=(32, NSG * (N // (NSG * RPS)), 1),
                        threadgroup=(32, NSG, 1),
                        output_shapes=[(N,)], output_dtypes=[mx.bfloat16])[0]
        # validate on replica 0
        ref = stock(0)
        out = run(0)
        mx.eval(ref, out)
        err = mx.abs(out.astype(mx.float32)
                     - ref.reshape(-1).astype(mx.float32)).max().item()
        if err > 0.05:
            print(f"p{PPT} rps{RPS} nsg{NSG}: numerics ERR {err:.4f}, skipped")
            continue
        cands[f"p{PPT} rps{RPS} nsg{NSG}"] = run

    print("pinning clocks...")
    pin_clocks()
    res = timed(cands)
    print(f"\ncache-busted, interleaved  ({nrep} replicas of "
          f"{nb2/1e6:.1f} MB):")
    print(f"{'variant':22s} {'us':>7s} {'GB/s':>7s} {'%peak':>6s} {'Gelem/s':>8s}")
    for name, t in res.items():
        print(f"{name:22s} {t*1e6:7.1f} {nb2/t/1e9:7.1f} "
              f"{100*nb2/t/1e9/PEAK_GBS:5.1f}% {N*K/t/1e9:8.1f}")

    del reps2, cands
    mx.clear_cache()

    # honest per-bit stock table
    print("\nstock qmv per-bit (cache-busted):")
    for bits in (2, 3, 4, 6, 8):
        reps = make_replicas(bits)
        nrepb = len(reps)
        nb = N * K * bits / 8 + 2 * (N * K // GS) * 2

        def sb(i, reps=reps, nrepb=nrepb, bits=bits):
            wq, s, b = reps[i % nrepb]
            return mx.quantized_matmul(x, wq, scales=s, biases=b,
                                       transpose=True, group_size=GS, bits=bits)
        pin_clocks(0.5)
        t = timed({"s": sb}, rounds=8)["s"]
        print(f"  {bits}-bit: {t*1e6:7.1f} us  {nb/t/1e9:6.1f} GB/s "
              f"({100*nb/t/1e9/PEAK_GBS:4.1f}%)  {N*K/t/1e9:6.1f} Gelem/s")
        del reps
        mx.clear_cache()


if __name__ == "__main__":
    main()
