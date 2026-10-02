"""The three decoders that read their quant bytes as aligned 32-bit words (IQ4_XS, Q4_K,
Q5_K) against the byte-load versions they replaced, in chad's own GEMV body: GB/s and a
bit-identity check of the bf16 output. Same values, same operation order, so the outputs
must be identical; a `False` here is a regression.

    uv run python benchmarks/gguf_decoders/load_width_race.py
"""
from __future__ import annotations

import os
import sys
from typing import cast

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx  # noqa: E402
from _common import chain, copies_for, synth  # noqa: E402
from gguf import GGMLQuantizationType as Q  # noqa: E402

from chad import mlx_gguf  # noqa: E402

NR, NSG = 2, 4

# The decoders as they were before the word loads: one byte load per value.
BYTE_DECODERS = r"""
static inline void deq32_iq4_xs_bytes(device const uint8_t* row, uint c, thread float* v) {
    device const uint8_t* b = row + 136u * (c >> 3);
    const uint j = c & 7u;
    const uint lo = (b[4 + (j >> 1)] >> (4u * (j & 1u))) & 15u;
    const uint hi = (gg_u16(b + 2) >> (2u * j)) & 3u;
    const float dl = gg_f16(b) * float(int(lo | (hi << 4)) - 32);
    device const uint8_t* q = b + 8 + 16 * j;
    for (int i = 0; i < 16; ++i) {
        v[i] = dl * float(kvalues_iq4nl[q[i] & 15u]);
        v[i + 16] = dl * float(kvalues_iq4nl[q[i] >> 4]);
    }
}
static inline void deq32_q4_k_bytes(device const uint8_t* row, uint c, thread float* v) {
    device const uint8_t* b = row + 144u * (c >> 3);
    const uint j = c & 7u;
    uint sc, m;
    gg_scale_min_k4(b + 4, j, sc, m);
    const float dl = gg_f16(b) * float(sc);
    const float ml = gg_f16(b + 2) * float(m);
    device const uint8_t* q = b + 16 + (j >> 1) * 32;
    const uint sh = 4u * (j & 1u);
    for (int i = 0; i < 32; ++i) v[i] = dl * float((q[i] >> sh) & 15u) - ml;
}
static inline void deq32_q5_k_bytes(device const uint8_t* row, uint c, thread float* v) {
    device const uint8_t* b = row + 176u * (c >> 3);
    const uint j = c & 7u;
    uint sc, m;
    gg_scale_min_k4(b + 4, j, sc, m);
    const float dl = gg_f16(b) * float(sc);
    const float ml = gg_f16(b + 2) * float(m);
    device const uint8_t* qh = b + 16;
    device const uint8_t* q = b + 48 + (j >> 1) * 32;
    const uint sh = 4u * (j & 1u);
    for (int i = 0; i < 32; ++i) {
        const uint x = ((q[i] >> sh) & 15u) | (((qh[i] >> j) & 1u) << 4);
        v[i] = dl * float(x) - ml;
    }
}
"""
_kernels: dict[str, mlx_gguf._MetalKernel] = {}


def byte_matmul(fmt: str, x, w, qtype: int, k: int):
    """chad's M=1 GEMV body over the byte-load decoder of `fmt`."""
    name = f"deq32_{fmt.lower()}_bytes"
    kern = _kernels.get(name)
    if kern is None:
        # SAFETY: the stub types metal_kernel's result as a bare object; it is the
        # keyword-called kernel _MetalKernel describes (the same cast mlx_gguf makes).
        kern = cast(mlx_gguf._MetalKernel, mx.fast.metal_kernel(
            name=f"race_{name}", input_names=["x", "w"], output_names=["out"],
            source=mlx_gguf._MV_SRC.replace("DEQ", name),
            header=mlx_gguf.metal_header() + BYTE_DECODERS))
        _kernels[name] = kern
    n = w.shape[0]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    (out,) = kern(
        inputs=[x2, w],
        template=[("T", x.dtype), ("MM", m), ("NR", NR), ("NSG", NSG), ("KD", k), ("ND", n),
                  ("ROWB", mlx_gguf.row_bytes(qtype, k))],
        output_shapes=[(m, n)], output_dtypes=[x.dtype],
        grid=(32 * NSG, (n + NR * NSG - 1) // (NR * NSG), 1), threadgroup=(32 * NSG, 1, 1))
    return out.reshape(*x.shape[:-1], n)


def main() -> None:
    for name, k, n in (("gate/up 5120->17408", 5120, 17408), ("down 17408->5120", 17408, 5120),
                       ("lm_head 5120->248320", 5120, 248320)):
        x0 = mx.random.normal((1, k), dtype=mx.bfloat16)
        mx.eval(x0)
        for fmt in ("IQ4_XS", "Q4_K", "Q5_K"):
            if fmt == "Q4_K" and n > 100000:
                continue
            qt = int(Q[fmt])
            rb = mlx_gguf.row_bytes(qt, k)
            ws = [synth(n, k, qt) for _ in range(copies_for(n, rb))]

            def words(x, t, ws=ws, qt=qt, k=k):
                return mlx_gguf.matmul(x, ws[t % len(ws)], qt, k)

            def bytes_(x, t, ws=ws, qt=qt, k=k, fmt=fmt):
                return byte_matmul(fmt, x, ws[t % len(ws)], qt, k)
            yw, yb = words(x0, 0), bytes_(x0, 0)
            mx.eval(yw, yb)
            same = bool(mx.array_equal(yw.view(mx.uint16), yb.view(mx.uint16)).item())
            mb, mw = chain(bytes_, x0), chain(words, x0)
            print(f"{name:22s} {fmt:7s} bytes {n * rb / mb / 1e6:4.0f} GB/s  words "
                  f"{n * rb / mw / 1e6:4.0f} GB/s  {100 * (mb - mw) / mb:+.0f}%  bit-identical {same}")
            mx.clear_cache()


if __name__ == "__main__":
    main()
