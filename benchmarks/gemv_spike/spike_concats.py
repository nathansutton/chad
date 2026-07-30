"""GEMV-fusion spike, part 8: router+seg and q|k|v concats (serialized).

Both are the proven mlx_fastpath weight-concat pattern (bit-exact per row).
Their value is dispatch removal in the dependent decode chain, so measurement
here is serialized: an 8-step dependent chain per timed call, replica-cycled.

  - router (2048->256 @8) + shared_expert_gate (2048->1 @8) -> one 257-row
    qmv (also tested zero-padded to 264 rows to stay on mlx's fast-qmv path,
    which needs rows % 8 == 0).
  - attn q (2048->4096) + k (2048->512) + v (2048->512), all @6
    -> one 5120-row qmv.

Run (from benchmarks/gemv_spike):
  uv run --project ../.. python spike_concats.py
"""
import mlx.core as mx

from spike_qmv_tune2 import pin_clocks, timed

GS = 64
PEAK_GBS = 273.0
HID = 2048
CHAIN = 8


def qz(n, k, bits, pad_to=None):
    w = mx.random.normal((n, k)).astype(mx.bfloat16)
    if pad_to is not None and pad_to > n:
        w = mx.concatenate([w, mx.zeros((pad_to - n, k), dtype=w.dtype)])
    q = mx.quantize(w, group_size=GS, bits=bits)
    mx.eval(q)
    return q


def qmv(x, q, bits):
    return mx.quantized_matmul(x, q[0], scales=q[1], biases=q[2],
                               transpose=True, group_size=GS, bits=bits)


def chained(step, x, nrep):
    def run(i):
        cur = x
        outs = []
        for j in range(CHAIN):
            y = step(cur, (i * CHAIN + j) % nrep)
            outs.append(y)
            cur = (x + 0.0 * mx.sum(y)).astype(x.dtype)
        return outs
    return run


def report(title, res, base_key):
    print(f"\n{title}")
    tb = res[base_key] / CHAIN
    for name, t in res.items():
        t /= CHAIN
        d = "" if name == base_key else \
            f"  ({(tb-t)*1e6:+.1f} us/step vs stock, x40 = " \
            f"{(tb-t)*40*1e3:.2f} ms/token)" if "router" in title.lower() else \
            f"  ({(tb-t)*1e6:+.1f} us/step vs stock, x10 = " \
            f"{(tb-t)*10*1e3:.2f} ms/token)"
        print(f"  {name:20s} {t*1e6:7.1f} us/step{d}")


def main():
    mx.random.seed(10)
    x = mx.random.normal((1, 1, HID)).astype(mx.bfloat16)

    # ---- router + seg (both 8-bit)
    nrep = 40  # small weights; replicas mostly bust locality of scales
    r_sep = [(qz(256, HID, 8), qz(1, HID, 8)) for _ in range(nrep)]
    r_cat257 = [qz(257, HID, 8) for _ in range(nrep)]
    r_cat264 = [qz(257, HID, 8, pad_to=264) for _ in range(nrep)]
    mx.clear_cache()

    def sep(cur, i):
        a, bq = r_sep[i]
        return mx.concatenate([qmv(cur, a, 8), qmv(cur, bq, 8)], axis=-1)

    def cat257(cur, i):
        return qmv(cur, r_cat257[i], 8)

    def cat264(cur, i):
        return qmv(cur, r_cat264[i], 8)

    print("pinning clocks...")
    pin_clocks()
    res = timed({"stock (2 qmv)": chained(sep, x, nrep),
                 "concat 257": chained(cat257, x, nrep),
                 "concat 264 (pad)": chained(cat264, x, nrep)},
                rounds=10, inner=4)
    report("router|seg (8-bit):", res, "stock (2 qmv)")
    del r_sep, r_cat257, r_cat264
    mx.clear_cache()

    # ---- q|k|v (6-bit): 4096+512+512 vs 5120
    per_mb = 5120 * HID * (6 / 8 + 4 / GS) / 1e6
    nrep = max(3, int(384 / per_mb) + 1)
    qkv_sep = [(qz(4096, HID, 6), qz(512, HID, 6), qz(512, HID, 6))
               for _ in range(nrep)]
    qkv_cat = [qz(5120, HID, 6) for _ in range(nrep)]
    mx.clear_cache()

    def sep3(cur, i):
        q, k, v = qkv_sep[i]
        return mx.concatenate(
            [qmv(cur, q, 6), qmv(cur, k, 6), qmv(cur, v, 6)], axis=-1)

    def cat1(cur, i):
        return qmv(cur, qkv_cat[i], 6)

    pin_clocks(1.0)
    res = timed({"stock (3 qmv)": chained(sep3, x, nrep),
                 "concat 5120": chained(cat1, x, nrep)},
                rounds=10, inner=4)
    report("attn q|k|v (6-bit):", res, "stock (3 qmv)")


if __name__ == "__main__":
    main()
