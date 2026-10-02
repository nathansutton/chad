"""Old vs new GGUF kernels inside the real decode step, one model load: the old
`mlx_gguf.py` is read from a git revision, its kernels built in-process and swapped into
chad's kernel cache per arm. The compiled layer bodies are rebuilt per arm — they capture
the kernel primitives when first traced, so a swap without the rebuild changes only the
attention and lm_head GEMVs. Reports the engine's serial step and a pipelined fixed-token
forward (the GPU floor), or with `--drafted` the DFlash2 path, and checks that greedy
text is identical across arms.

    CHAD_NO_DFLASH=1 uv run python benchmarks/gguf_decoders/decode_ab.py --old-rev main
    uv run python benchmarks/gguf_decoders/decode_ab.py --old-rev main --drafted
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

import mlx.core as mx  # noqa: E402

from chad import mlx_gguf  # noqa: E402
from chad.bench import _bench_engine, _build_prompt, _pick_model  # noqa: E402
from chad.mlx_fastpath import _gdn_body, _gguf_gdn_projections, _gguf_mlp_body  # noqa: E402


def load_old_module(rev: str):
    src = subprocess.run(["git", "-C", ROOT, "show", f"{rev}:src/chad/mlx_gguf.py"],
                         capture_output=True, text=True, check=True).stdout
    path = os.path.join(HERE, "_mlx_gguf_old.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("chad.mlx_gguf_old", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # SAFETY: a spec built from a file path has a loader
    os.remove(path)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-rev", default="HEAD~1", help="git revision holding the old mlx_gguf.py")
    ap.add_argument("--drafted", action="store_true", help="measure with the drafter on")
    ap.add_argument("--tokens", type=int, default=96)
    a = ap.parse_args()
    old = load_old_module(a.old_rev)

    eng = _bench_engine(_pick_model()[0])
    eng.load()
    eng.temp = 0.0
    layers = eng.model.language_model.model.layers
    gdn = [layer for layer in layers if layer.is_linear]
    prompt = _build_prompt(eng.tok, 512)
    eng.generate(prompt, max_tokens=8)   # every kernel the model uses now exists
    new_kernels = dict(mlx_gguf._kernels)
    sources = {"mv": (old._MV_SRC, ["x", "w"]), "mm": (old._MM_SRC, ["x", "w"]),
               "deq": (old._SRC, ["w"])}
    old_kernels = {key: old._kernel(key[0], sources[key[0]][0], sources[key[0]][1], key[1])
                   for key in new_kernels}
    print(f"drafter {'on' if eng._dflash is not None else 'off'}; old kernels from {a.old_rev}",
          flush=True)

    def arm(kernels) -> None:
        mlx_gguf._kernels.clear()
        mlx_gguf._kernels.update(kernels)
        for layer in layers:
            layer._mlp_fast = mx.compile(_gguf_mlp_body(layer))
        for layer in gdn:
            layer._gdn_fast = mx.compile(_gdn_body(layer, *_gguf_gdn_projections(layer.linear_attn)))

    fixed = mx.array([prompt[-1]], dtype=mx.uint32)

    def step(y):
        logits = eng.model(y[None], cache=eng._cache)[:, -1]
        return (mx.argmax(logits, axis=-1) * 0 + fixed).astype(mx.uint32)

    def pipelined() -> float:
        y = step(fixed)
        mx.async_eval(y)
        for _ in range(4):
            ny = step(y)
            mx.async_eval(ny)
            mx.eval(y)
            y = ny
        t0 = time.perf_counter()
        for _ in range(a.tokens):
            ny = step(y)
            mx.async_eval(ny)
            mx.eval(y)
            y = ny
        mx.eval(y)
        return (time.perf_counter() - t0) / a.tokens * 1e3

    def engine() -> tuple[float, str]:
        text, s = eng.generate(prompt, max_tokens=a.tokens if not a.drafted else 2 * a.tokens)
        return 1e3 / s.tok_per_s, text

    results: dict[tuple[str, str], float] = {}
    texts: dict[str, str] = {}
    for _ in range(2):
        for name, kernels in (("old", old_kernels), ("new", new_kernels)):
            arm(kernels)
            ms, text = engine()
            results[(name, "engine")] = min(results.get((name, "engine"), 1e9), ms)
            texts[name] = text
            if not a.drafted:
                results[(name, "pipelined")] = min(results.get((name, "pipelined"), 1e9), pipelined())
    for (name, kind), ms in sorted(results.items()):
        print(f"  {name:4s} {kind:10s} {ms:7.2f} ms/token ({1e3 / ms:5.1f} tok/s)")
    o, n = results[("old", "engine")], results[("new", "engine")]
    print(f"engine step: old {o:.2f} -> new {n:.2f} ms/token = {100 * (o - n) / o:+.1f}%; "
          f"greedy text {'identical' if texts['old'] == texts['new'] else 'DIFFERS'}")


if __name__ == "__main__":
    main()
