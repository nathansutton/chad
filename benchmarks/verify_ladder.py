"""Verify-width ladder: what one main-model forward of width M costs, in serial steps.

The speculative schedule (`mlx_dflash.WidthPolicy`) prices a round from a seed ladder
T(d), and the ladder's SHAPE depends on the weights' bit width and on where the small-M
MMA kernel takes over from stock `quantized_matmul` — so it has to be re-measured per
checkpoint, not carried from one quant to the next. This measures it directly: ONE model
load, the engine's own cache (8-bit KV when that is what the engine runs), a real context
in front, then a timed forward at every width.

    uv run python benchmarks/verify_ladder.py                 # the shipped model
    uv run python benchmarks/verify_ladder.py --ctx 8192 --json ladder.json

Widths past 8 are there for the tiled-MMA path (warm tails, tool-result suffixes, wide
lookup verifies). Reported in ms and in units of the M=1 step. The drafter's own forward
is NOT in these numbers; `--drafter` adds its per-round cost so the printed seed row is
the full round, which is what BLOCK_ROUND_COSTS holds.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

WIDTHS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 24, 32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--ctx", type=int, default=2048, help="context tokens in front")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import mlx.core as mx

    from chad.cli import _pick_model
    from chad.engine import Engine

    eng = Engine(model_id=a.model or _pick_model()[0], cache_dir=None)
    print(f"load {eng.load():.1f}s", flush=True)
    model, cache = eng.model, eng._cache
    vocab = int(model.language_model.args.vocab_size)
    mx.random.seed(0)
    ids = mx.random.randint(1000, min(vocab, 100000), (1, a.ctx + 4096))
    pos = 0
    while pos < a.ctx:                      # prefill in the engine-sized chunks
        n = min(512, a.ctx - pos)
        mx.eval(model(ids[:, pos:pos + n], cache=cache))
        pos += n
    rows = {}
    for M in WIDTHS:
        times = []
        for r in range(a.reps + 2):
            x = ids[:, pos:pos + M]
            t0 = time.perf_counter()
            mx.eval(model(x, cache=cache))
            times.append(time.perf_counter() - t0)
            pos += M
        rows[M] = statistics.median(times[2:]) * 1e3   # first two: kernel builds
    base = rows[1]
    print(f"ctx {a.ctx}  serial step {base:.1f} ms ({1e3 / base:.1f} tok/s)")
    for M, ms in rows.items():
        print(f"  M={M:<3d} {ms:7.1f} ms  {ms / base:5.2f} steps  {ms / M:6.1f} ms/row")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"ctx": a.ctx, "ms": rows}, f, indent=1)


if __name__ == "__main__":
    main()
