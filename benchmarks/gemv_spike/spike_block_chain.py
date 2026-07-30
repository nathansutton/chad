"""GEMV-fusion spike, part 9: integrated MoE block, serialized, + projection.

The number that actually lands: a full MoE block step
  stock: rms_norm -> router qmm -> softmax/top-8/norm -> gather_qmm x2 +
         silu*mul + weighted sum -> shared expert (3 qmm + seg + sigmoid*) ->
         residual   (exact _moe_body structure, compiled)
  fused: rms_norm -> router|seg qmv (264 rows, padded) -> softmax/top-8/norm
         -> A+ -> B+ (shared expert, scores, sigmoid(seg), residual all in
         the two custom dispatches)
measured as an 8-step dependent chain (decode reality), replica-cycled,
clock-pinned, interleaved. Ends with the per-token projection for the 35B.

Run (from benchmarks/gemv_spike):
  uv run --project ../.. python spike_block_chain.py
"""
import mlx.core as mx
import mlx.nn as nn

import spike_fused_expert3 as sf
from spike_qmv_tune2 import pin_clocks, timed

GS = 64
PEAK_GBS = 273.0
HID = 2048
E = 256
TOP_K = 8
INTER = 512
NGU = 2 * INTER
CHAIN = 8
BASELINE_MS = 14.4      # measured 35B decode @8k, 69.6 tok/s
N_LAYERS = 40


def qz(n, k, bits, pad_to=None):
    w = mx.random.normal((n, k)).astype(mx.bfloat16) * 0.3
    if pad_to is not None and pad_to > n:
        w = mx.concatenate([w, mx.zeros((pad_to - n, k), dtype=w.dtype)])
    q = mx.quantize(w, group_size=GS, bits=bits)
    mx.eval(q)
    return q


def qmm(x, t, bits):
    return mx.quantized_matmul(x, t[0], scales=t[1], biases=t[2],
                               transpose=True, group_size=GS, bits=bits)


def make_replica():
    w = sf.make_weights()             # experts 2-bit + shared 6-bit
    router = qz(256, HID, 8)
    segq = qz(1, HID, 8)
    # fused router|seg: rows 0..255 router, 256 seg, 257..263 zero pad
    rw = mx.concatenate([router[0], segq[0],
                         mx.zeros((7,) + router[0].shape[1:],
                                  dtype=router[0].dtype)])
    rs = mx.concatenate([router[1], segq[1],
                         mx.zeros((7,) + router[1].shape[1:],
                                  dtype=router[1].dtype)])
    rb = mx.concatenate([router[2], segq[2],
                         mx.zeros((7,) + router[2].shape[1:],
                                  dtype=router[2].dtype)])
    mx.eval(rw, rs, rb)
    return w, router, segq, (rw, rs, rb)


def stock_block(h, w, router, segq, ln_w):
    """Exact _moe_body structure (norm_topk_prob=True)."""
    x = mx.fast.rms_norm(h, ln_w, 1e-6)
    gates = mx.softmax(qmm(x, router, 8), axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-TOP_K, axis=-1)[..., -TOP_K:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)

    guw2, gus2, gub2, dw2, ds2, db2 = w[:6]
    guw6, gus6, gub6 = w[6:9]
    dnw6, dns6, dnb6 = w[9:12]
    xe = mx.expand_dims(x, (-2, -3))
    gu = mx.gather_qmm(xe, guw2, gus2, gub2, rhs_indices=inds, transpose=True,
                       group_size=GS, bits=2, sorted_indices=False)
    xg, xu = mx.split(gu, 2, axis=-1)
    hh = nn.silu(xg) * xu
    yy = mx.gather_qmm(hh, dw2, ds2, db2, rhs_indices=inds, transpose=True,
                       group_size=GS, bits=2, sorted_indices=False)
    y = (yy.squeeze(-2) * scores[..., None]).sum(axis=-2)

    sh_g = (guw6[:INTER], gus6[:INTER], gub6[:INTER])
    sh_u = (guw6[INTER:], gus6[INTER:], gub6[INTER:])
    sh = qmm(nn.silu(qmm(x, sh_g, 6)) * qmm(x, sh_u, 6), (dnw6, dns6, dnb6), 6)
    seg = qmm(x, segq, 8)
    return h + y + mx.sigmoid(seg) * sh


def fused_block(ka, kb, h, w, rcat, ln_w):
    x = mx.fast.rms_norm(h, ln_w, 1e-6)
    rg = qmm(x, rcat, 8)
    gates = mx.softmax(rg[..., :256], axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-TOP_K, axis=-1)[..., -TOP_K:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    seg = rg[..., 256]

    guw2, gus2, gub2, dw2, ds2, db2 = w[:6]
    guw6, gus6, gub6 = w[6:9]
    dnw6, dns6, dnb6 = w[9:12]
    hmid = ka(inputs=[x.reshape(-1), guw2, gus2, gub2, guw6, gus6, gub6,
                      inds.reshape(-1).astype(mx.uint32)],
              template=[("T", mx.bfloat16)],
              grid=(32, 2 * (8 * 64 + 64), 1), threadgroup=(32, 2, 1),
              output_shapes=[(TOP_K + 1, INTER)],
              output_dtypes=[mx.bfloat16])[0]
    y = kb(inputs=[hmid.reshape(-1), dw2, ds2, db2, dnw6, dns6, dnb6,
                   inds.reshape(-1).astype(mx.uint32),
                   scores.astype(mx.float32).reshape(-1),
                   seg.astype(mx.float32).reshape(-1), h.reshape(-1)],
           template=[("T", mx.bfloat16)],
           grid=(32, 2 * (2048 // (2 * 8)), 1), threadgroup=(32, 2, 1),
           output_shapes=[(HID,)], output_dtypes=[mx.bfloat16])[0]
    return y.reshape(1, 1, HID)


def main():
    mx.random.seed(11)
    x0 = mx.random.normal((1, 1, HID)).astype(mx.bfloat16)
    ln_w = mx.ones((HID,), dtype=mx.bfloat16)

    per_mb = (E * (NGU * HID + HID * INTER) * (2 / 8 + 4 / GS)) / 1e6
    nrep = max(3, int(384 / per_mb) + 1)
    reps = [make_replica() for _ in range(nrep)]
    mx.clear_cache()

    ka, kb = sf.build()
    scl = mx.compile(
        lambda h, w, router, segq: stock_block(h, w, router, segq, ln_w))

    # numerics: fused vs stock full block
    w, router, segq, rcat = reps[0]
    ref = scl(x0, w, router, segq)
    out = fused_block(ka, kb, x0, w, rcat, ln_w)
    mx.eval(ref, out)
    r32 = ref.reshape(-1).astype(mx.float32)
    err = mx.abs(out.reshape(-1).astype(mx.float32) - r32).max().item()
    scale = mx.abs(r32).max().item()
    print(f"numerics (full block): max abs {err:.4f} on |y|max {scale:.1f} "
          f"({err/scale:.2e} rel)")
    # routing parity: same experts chosen (indices via identical graph ops)
    print("routing inputs identical:",
          bool((qmm(mx.fast.rms_norm(x0, ln_w, 1e-6), router, 8)
                - qmm(mx.fast.rms_norm(x0, ln_w, 1e-6), rcat, 8)[..., :256])
               .abs().max().item() == 0.0))

    def chain(step):
        def run(i):
            cur = x0
            outs = []
            for j in range(CHAIN):
                y = step(cur, (i * CHAIN + j) % nrep)
                outs.append(y)
                cur = (x0 + 0.0 * mx.sum(y)).astype(x0.dtype)
            return outs
        return run

    def stock_step(cur, i):
        w, router, segq, _ = reps[i]
        return scl(cur, w, router, segq)

    def fused_step(cur, i):
        w, _, _, rcat = reps[i]
        return fused_block(ka, kb, cur, w, rcat, ln_w)

    print(f"pinning clocks... ({nrep} replicas)")
    pin_clocks()
    res = timed({"stock block": chain(stock_step),
                 "fused block": chain(fused_step)}, rounds=12, inner=3)
    ts = res["stock block"] / CHAIN
    tf = res["fused block"] / CHAIN
    print(f"\nstock MoE block  {ts*1e6:7.1f} us/step")
    print(f"fused MoE block  {tf*1e6:7.1f} us/step   ({ts/tf:.2f}x)")

    d_moe = (ts - tf) * N_LAYERS
    d_qkv = 0.03e-3    # part 8, x10 layers
    for name, d in [("fused MoE only", d_moe), ("+ qkv concat", d_moe + d_qkv)]:
        new_ms = BASELINE_MS - d * 1e3
        print(f"{name:16s}: {d*1e3:5.2f} ms/token -> {new_ms:.2f} ms "
              f"= {1000/new_ms:.1f} tok/s ({1000/new_ms/ (1000/BASELINE_MS) - 1:+.1%})")


if __name__ == "__main__":
    main()
