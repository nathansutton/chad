"""Shared pieces of the GGUF decoder benchmarks: dependent-chain timing the way chad's
own kernel probes time (nothing cache-resident, the critical path measured), and
synthetic weight rows whose block scales are sane f16 so outputs are finite."""
from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

from chad import mlx_gguf

CHAIN, EVALS = 12, 6
SHAPES = [("gate/up 5120->17408", 5120, 17408), ("down 17408->5120", 17408, 5120),
          ("qkvz 5120->16384", 5120, 16384), ("o/out 6144->5120", 6144, 5120)]


def chain(step, x0) -> float:
    """Median ms of one step over dependent chains of CHAIN calls (first eval warms)."""
    times = []
    for _ in range(EVALS):
        x = x0
        t0 = time.perf_counter()
        for t in range(CHAIN):
            y = step(x, t)
            x = x0 + mx.mean(y).astype(x0.dtype) * 1e-20
        mx.eval(y, x)
        times.append((time.perf_counter() - t0) / CHAIN)
    rest = sorted(times[1:])
    return rest[len(rest) // 2] * 1e3


def synth(n: int, k: int, qtype: int):
    """Random GGUF rows of `qtype` with every block's leading f16 set to 0.0625."""
    rb = mlx_gguf.row_bytes(qtype, k)
    bb = mlx_gguf.FORMATS[qtype].block_bytes
    a = np.random.randint(0, 256, (n, rb), dtype=np.uint8)
    a[:, 0::bb] = 0x00
    a[:, 1::bb] = 0x2C
    w = mx.array(a)
    mx.eval(w)
    return w


def copies_for(n: int, rb: int, budget: int = 1_200_000_000) -> int:
    return max(2, min(6, budget // (n * rb)))
