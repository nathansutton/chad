"""Final adoption gate for mlx_moe_fused on the real 35B (one load).

  1. greedy parity: 3 prompts x 200 tokens, fused vs compiled body
  2. decode-path NLL: teacher-forced S=1 stepping over real text through the
     decode path both ways — the quality gate (prefill logits can't see the
     kernels, so PPL must be measured through the decode step itself)
  3. decode tok/s: short and ~8k ctx, body/fused/body bracketed

Run:  uv run python benchmarks/gemv_spike/gate_real35b_v2.py
"""
import math
import time

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models import cache as cache_utils

from chad import mlx_fastpath

MODEL = "nathansutton/Ornith-1.0-35B-UD-Q2_K_XL-MLX"

PROMPTS = [
    "def quicksort(arr):\n",
    "The key difference between a mutex and a semaphore is",
    "import numpy as np\n\n# Compute the eigenvalues of a symmetric matrix\n",
]

NLL_TEXT_PATH = "README.md"
NLL_TOKENS = 900
NLL_PREFILL = 64


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


def decode_nll(model, ids):
    """Teacher-forced S=1 stepping (the path the kernels live on)."""
    cache = cache_utils.make_prompt_cache(model)
    logits = model(mx.array(ids[:NLL_PREFILL], dtype=mx.uint32)[None],
                   cache=cache)
    mx.eval([c.state for c in cache])
    nll, count = 0.0, 0
    for i in range(NLL_PREFILL, len(ids) - 1):
        logits = model(mx.array([ids[i]], dtype=mx.uint32)[None], cache=cache)
        l32 = logits[0, -1].astype(mx.float32)
        nll -= float(l32[ids[i + 1]] - mx.logsumexp(l32))
        count += 1
    return nll / count


def main():
    t0 = time.time()
    model, tok = load(MODEL)
    assert mlx_fastpath.install(model) is True
    assert getattr(model.language_model.model.layers[0].mlp,
                   "_fused_step", None) is not None
    print(f"loaded + installed in {time.time()-t0:.0f}s")

    for i, p in enumerate(PROMPTS):
        set_path(model, "fused")
        a, _ = gen_tokens(model, tok, p, 200)
        set_path(model, "body")
        b, _ = gen_tokens(model, tok, p, 200)
        d = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y),
                 min(len(a), len(b)))
        status = "IDENTICAL" if a == b else f"DIVERGED at token {d}"
        print(f"parity prompt {i}: {status} ({len(a)}/{len(b)} toks)")

    ids = tok.encode(open(NLL_TEXT_PATH).read())[:NLL_TOKENS]
    print(f"NLL corpus: {len(ids)} tokens of {NLL_TEXT_PATH}")
    set_path(model, "body")
    nb = decode_nll(model, ids)
    set_path(model, "fused")
    nf = decode_nll(model, ids)
    print(f"decode NLL: body {nb:.5f}  fused {nf:.5f}  "
          f"dPPL {100*(math.exp(nf)/math.exp(nb)-1):+.3f}%")

    long_prompt = ("# " + "The quick brown fox jumps over the lazy dog. "
                   * 700 + "\ndef main():\n")
    for label, prompt, n in [("short", PROMPTS[0], 256),
                             ("~8k ctx", long_prompt, 192)]:
        results = []
        for which in ("body", "fused", "body"):
            set_path(model, which)
            _, tps = gen_tokens(model, tok, prompt, n)
            results.append(tps)
        b1, f, b2 = results
        print(f"{label}: body {b1:.1f}/{b2:.1f}, fused {f:.1f} tok/s "
              f"-> {f/((b1+b2)/2)-1:+.1%}")
    print(f"peak memory {mx.get_peak_memory()/1e9:.1f} GB")


if __name__ == "__main__":
    main()
