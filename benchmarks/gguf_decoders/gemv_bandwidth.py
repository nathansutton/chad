"""Achieved bandwidth of chad's M=1 GGUF GEMV per block format on the 27B projection
shapes, with Q8_0 (the cheapest decode) and MLX's affine kernels as the ceiling of what
the kernel body and the chip can stream. Synthetic rows: the timing is data-independent.

    uv run python benchmarks/gguf_decoders/gemv_bandwidth.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx  # noqa: E402
from _common import SHAPES, chain, copies_for, synth  # noqa: E402
from gguf import GGMLQuantizationType as Q  # noqa: E402

from chad import mlx_gguf  # noqa: E402

FORMATS = [q for q in ("IQ4_XS", "IQ3_S", "Q5_K", "IQ3_XXS", "Q3_K", "Q4_K", "IQ2_S", "Q6_K",
                       "IQ2_XS", "Q8_0") if int(Q[q]) in mlx_gguf.FORMATS]


def main() -> None:
    print(f"{'shape':22s} {'format':8s} {'bytes/row':>9s} {'ms':>7s} {'GB/s':>6s}")
    for name, k, n in SHAPES:
        x0 = mx.random.normal((1, k), dtype=mx.bfloat16)
        mx.eval(x0)
        for fmt in FORMATS:
            qt = int(Q[fmt])
            rb = mlx_gguf.row_bytes(qt, k)
            ws = [synth(n, k, qt) for _ in range(copies_for(n, rb))]

            def step(x, t, ws=ws, qt=qt, k=k):
                return mlx_gguf.matmul(x, ws[t % len(ws)], qt, k)
            mx.eval(step(x0, 0))
            ms = chain(step, x0)
            print(f"{name:22s} {fmt:8s} {rb:9d} {ms:7.3f} {n * rb / ms / 1e6:6.0f}")
            mx.clear_cache()
        for bits in (3, 4):
            ws = [mx.quantize(mx.random.normal((n, k), dtype=mx.bfloat16) * 0.02, group_size=64,
                              bits=bits) for _ in range(4)]
            mx.eval(*[a for w in ws for a in w])
            nbytes = sum(a.nbytes for a in ws[0])

            def step(x, t, ws=ws, bits=bits):
                return mx.quantized_matmul(x, *ws[t % 4], transpose=True, group_size=64, bits=bits)
            mx.eval(step(x0, 0))
            ms = chain(step, x0)
            print(f"{name:22s} {'affine' + str(bits):8s} {nbytes // n:9d} {ms:7.3f} {nbytes / ms / 1e6:6.0f}")
            mx.clear_cache()


if __name__ == "__main__":
    main()
