"""GEMV-fusion spike, part 1: where does the decode-step bandwidth go?

Builds every S=1 decode GEMV of the shipped Ornith 35B checkpoint at its EXACT
shape and quantization (synthetic weights via mx.quantize — no model load, no
GPU-panic-risk load cycling), then measures:

  1. throughput  — K independent copies of the op, one eval: achieved GB/s
                   with dispatch gaps overlappable (pure kernel efficiency).
  2. serialized  — a data-dependent chain of the same op: marginal cost per op
                   when nothing can overlap (decode reality). serialized minus
                   throughput time = per-dispatch gap cost.

Run:  uv run python benchmarks/gemv_spike/spike_ops.py

CAVEAT (discovered in part 3): these numbers are optimistic for any op whose
weights fit in the SLC (~tens of MB) — repeated calls on one weight buffer hit
cache, and GPU clocks ramp mid-script. Only the lm_head row (413 MB, uncache-
able) is DRAM-honest here. spike_qmv_tune2.py has the corrected harness
(replica cycling + clock pinning + interleaved A/B); its numbers supersede
these for absolute GB/s. The RELATIVE structure (which ops are small/serial-
ized) still stands.
"""
import json
import math
import time
from pathlib import Path

import mlx.core as mx

GS = 64
PEAK_GBS = 273.0  # M4 Pro 24GB

HID = 2048
# GDN geometry
KEY_DIM = 16 * 128
VAL_DIM = 32 * 128
CONV_DIM = 2 * KEY_DIM + VAL_DIM              # 8192
GDN_IN_OUT = CONV_DIM + VAL_DIM + 32 + 32     # 12352 (qkv|z|b|a fused)
# attention geometry
Q_OUT = 16 * 256
KV_OUT = 2 * 256
# MoE geometry
N_EXPERTS = 256
TOP_K = 8
E_INTER = 512
VOCAB = 248320


def qlinear(out_dim, in_dim, bits):
    w = mx.random.normal((out_dim, in_dim)).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=GS, bits=bits)
    mx.eval(wq, s, b)
    return wq, s, b, bits


def qbytes(out_dim, in_dim, bits):
    return out_dim * in_dim * bits / 8 + 2 * (out_dim * in_dim // GS) * 2


def qmv(x, m):
    wq, s, b, bits = m
    return mx.quantized_matmul(x, wq, scales=s, biases=b, transpose=True,
                               group_size=GS, bits=bits)


def bench(build_fn, iters=30, warmup=5):
    """build_fn() -> list of arrays to eval. Returns median seconds per call."""
    for _ in range(warmup):
        mx.eval(build_fn())
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(build_fn())
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def measure(name, op, x, nbytes, K=24, chain_map=None):
    """op: x -> y.  chain_map: y -> next x (for the serialized chain)."""
    t_ind = bench(lambda: [op(x) for _ in range(K)]) / K
    row = {"name": name, "bytes": nbytes, "us_tp": t_ind * 1e6,
           "gbs": nbytes / t_ind / 1e9}
    if chain_map is not None:
        def chained():
            cur = x
            outs = []
            for _ in range(K):
                y = op(cur)
                outs.append(y)
                cur = chain_map(y, x)
            return outs
        t_ser = bench(chained) / K
        row["us_ser"] = t_ser * 1e6
    return row


def main():
    mx.random.seed(0)
    x = mx.random.normal((1, 1, HID)).astype(mx.bfloat16)
    rows = []

    # ---- dispatch-gap probe: dependent rms_norm chain (negligible traffic)
    w_ln = mx.ones((HID,), dtype=mx.bfloat16)
    K = 200
    def norm_chain():
        cur = x
        for _ in range(K):
            cur = mx.fast.rms_norm(cur, w_ln, 1e-6)
        return [cur]
    t = bench(norm_chain) / K
    print(f"dispatch-gap probe (rms_norm 2048 dependent chain): "
          f"{t*1e6:.2f} us/op")

    # ---- the big GEMVs, throughput + serialized
    def sl(y, x0):  # map any output back to a (1,1,HID) input, cheap
        return x0 + 0.0 * y.reshape(1, 1, -1)[..., :HID]

    specs = [
        ("gdn_in_proj  2048->12352 @6", GDN_IN_OUT, HID, 6),
        ("gdn_out_proj 4096->2048  @6", HID, VAL_DIM, 6),
        ("attn_q_proj  2048->4096  @6", Q_OUT, HID, 6),
        ("attn_kv_proj 2048->512   @6", KV_OUT, HID, 6),
        ("attn_o_proj  4096->2048  @6", HID, Q_OUT, 6),
        ("router       2048->256   @8", N_EXPERTS, HID, 8),
        ("shared_g|u   2048->1024  @6", 2 * E_INTER, HID, 6),
        ("shared_down  512->2048   @6", HID, E_INTER, 6),
        ("lm_head      2048->248320@6", VOCAB, HID, 6),
    ]
    for name, out_d, in_d, bits in specs:
        m = qlinear(out_d, in_d, bits)
        xin = mx.random.normal((1, 1, in_d)).astype(mx.bfloat16)
        nb = qbytes(out_d, in_d, bits)
        chain = None if out_d == VOCAB else (lambda y, x0, d=in_d:
                 (x0 + 0.0 * mx.sum(y)).astype(mx.bfloat16))
        rows.append(measure(name, lambda a, m=m: qmv(a, m), xin, nb,
                            K=8 if out_d == VOCAB else 24, chain_map=chain))
        del m
        mx.clear_cache()

    # ---- expert gather_qmm (top-8 of 256, 2-bit): the MoE hot path
    for name, out_d, in_d in [("exp_gate|up  g_qmm 8x(2048->1024) @2", 2 * E_INTER, HID),
                              ("exp_down     g_qmm 8x(512->2048)  @2", HID, E_INTER)]:
        w = mx.random.normal((N_EXPERTS, out_d, in_d)).astype(mx.bfloat16)
        wq, s, b = mx.quantize(w, group_size=GS, bits=2)
        mx.eval(wq, s, b)
        del w
        inds = mx.array([[[3, 17, 42, 99, 120, 180, 201, 250]]], dtype=mx.uint32)
        xe = mx.random.normal((1, 1, 1, 1, in_d)).astype(mx.bfloat16)
        nb = TOP_K * qbytes(out_d, in_d, 2)

        def gop(a, wq=wq, s=s, b=b):
            return mx.gather_qmm(a, wq, s, b, rhs_indices=inds, transpose=True,
                                 group_size=GS, bits=2, sorted_indices=False)
        rows.append(measure(name, gop, xe, nb, K=24,
                            chain_map=lambda y, x0: (x0 + 0.0 * mx.sum(y)).astype(mx.bfloat16)))
        del wq, s, b
        mx.clear_cache()

    # ---- report
    print(f"\n{'op':38s} {'MB':>8s} {'us(tp)':>8s} {'us(ser)':>8s} "
          f"{'GB/s':>7s} {'%peak':>6s}")
    for r in rows:
        ser = f"{r['us_ser']:8.1f}" if "us_ser" in r else "       -"
        print(f"{r['name']:38s} {r['bytes']/1e6:8.2f} {r['us_tp']:8.1f} {ser} "
              f"{r['gbs']:7.1f} {100*r['gbs']/PEAK_GBS:5.1f}%")

    # ---- per-token roofline from these measurements
    by = {r["name"].split()[0]: r for r in rows}
    def us(k, mult, key="us_tp"):
        return by[k][key] * mult
    comp = {
        "gdn_in_proj x30": us("gdn_in_proj", 30),
        "gdn_out_proj x30": us("gdn_out_proj", 30),
        "attn q+kv+o x10": (us("attn_q_proj", 10) + us("attn_kv_proj", 20)
                            + us("attn_o_proj", 10)),
        "router+seg x40": us("router", 40) * 1.0,
        "shared g|u+down x40": us("shared_g|u", 40) + us("shared_down", 40),
        "experts g|u+down x40": us("exp_gate|up", 40) + us("exp_down", 40),
        "lm_head x1": us("lm_head", 1),
    }
    tot = sum(comp.values())
    print(f"\nGEMV-only per-token floor (excl. attention SDPA/GDN-state/norms):")
    for k, v in comp.items():
        print(f"  {k:24s} {v/1000:7.3f} ms")
    print(f"  {'TOTAL':24s} {tot/1000:7.3f} ms  -> {1e6/tot:.1f} tok/s ceiling")


if __name__ == "__main__":
    main()
