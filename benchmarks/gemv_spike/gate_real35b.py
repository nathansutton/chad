"""Adoption gate for mlx_moe_fused on the real 35B checkpoint.

ONE model load per process (GPU-panic rule: no load/teardown cycling). Both
paths live in the same process: `layer._moe_fast` is flipped between the
fused-kernel wrapper and the compiled stock-graph body.

  1. greedy parity: 3 prompts x 200 tokens, fused vs body token streams
  2. decode tok/s: short ctx and ~8k ctx, body/fused/body bracketed so DVFS
     drift shows up as disagreement between the two body runs

Run:  uv run python benchmarks/gemv_spike/gate_real35b.py
"""
import time

import mlx.core as mx
from mlx_lm import load, stream_generate

from chad import mlx_fastpath

MODEL = "nathansutton/Ornith-1.0-35B-UD-Q2_K_XL-MLX"

PROMPTS = [
    "def quicksort(arr):\n",
    "The key difference between a mutex and a semaphore is",
    "import numpy as np\n\n# Compute the eigenvalues of a symmetric matrix\n",
]


def set_path(model, which):
    for layer in model.language_model.model.layers:
        if which == "fused":
            f = layer.mlp._fused_step
            b = layer._moe_body_fast
            layer._moe_fast = (lambda h, f=f, b=b:
                               f(h) if h.size == h.shape[-1] else b(h))
        else:
            layer._moe_fast = layer._moe_body_fast


def gen_tokens(model, tok, prompt, n):
    toks, tps = [], 0.0
    for r in stream_generate(model, tok, prompt, max_tokens=n):
        toks.append(r.token)
        tps = r.generation_tps
    return toks, tps


def main():
    t0 = time.time()
    model, tok = load(MODEL)
    assert mlx_fastpath.install(model) is True
    lay0 = model.language_model.model.layers[0]
    assert getattr(lay0.mlp, "_fused_step", None) is not None, \
        "fused kernels did not engage on the real checkpoint"
    print(f"loaded + installed in {time.time()-t0:.0f}s "
          f"(peak {mx.get_peak_memory()/1e9:.1f} GB)")

    # ---- parity
    for i, p in enumerate(PROMPTS):
        set_path(model, "fused")
        a, _ = gen_tokens(model, tok, p, 200)
        set_path(model, "body")
        b, _ = gen_tokens(model, tok, p, 200)
        match = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y),
                     min(len(a), len(b)))
        status = "IDENTICAL" if a == b else f"DIVERGED at token {match}"
        print(f"prompt {i}: fused {len(a)} toks vs body {len(b)} toks -> "
              f"{status}")

    # ---- throughput, short ctx
    short = PROMPTS[0]
    long_prompt = ("# " + "The quick brown fox jumps over the lazy dog. "
                   * 700 + "\ndef main():\n")
    for label, prompt, n in [("short", short, 256), ("~8k ctx", long_prompt, 192)]:
        ntok = len(tok.encode(prompt))
        results = []
        for which in ("body", "fused", "body"):
            set_path(model, which)
            _, tps = gen_tokens(model, tok, prompt, n)
            results.append((which, tps))
        b1, f, b2 = results[0][1], results[1][1], results[2][1]
        base = (b1 + b2) / 2
        print(f"{label} ({ntok} prompt toks): body {b1:.1f}/{b2:.1f} tok/s, "
              f"fused {f:.1f} tok/s -> {f/base-1:+.1%}")
    print(f"peak memory {mx.get_peak_memory()/1e9:.1f} GB")


if __name__ == "__main__":
    main()
