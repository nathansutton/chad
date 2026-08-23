"""Is greedy speculative output token-identical to greedy serial output?

The docs claim it is, and the claim is worth auditing on the shipped model rather than
only on the tiny synthetic model in the unit tests. Speculation is exact by
CONSTRUCTION — a draft token is committed only where it equals the target's own argmax
— so a divergence is not the accept rule being wrong. It is the two paths computing
slightly different logits, with a near-tie landing on the other side. Three suspects,
and this script separates them:

  * `mlx_qmm_mma`, the small-M matmul that engages from SIX rows up — i.e. only inside
    a wide verify, never in serial decode. It is new, and it is the one that changed
    when the schedule started running the full block of 7.
  * the S>1 tier of the fused attention kernel (`mlx_qsdpa`), which serves the verify
    forward where serial decode runs the S=1 kernel. Same numerics class, not
    bit-identical to it.
  * the harness, if the arms do not start from identical cache state. Every run here
    does a FULL fresh prefill for exactly that reason: sharing a warm cache between
    arms (rewinding instead of re-prefilling) makes the comparison meaningless, and
    shows up as a divergence at generated-token 0 — which says something about the
    rewind, not about speculation.

All arms run under ONE model load. That is not just for speed: repeatedly loading and
tearing down a 12 GB model on a 24 GB Mac has panicked this class of machine, and both
kernel switches can be flipped in-process anyway (`config.flag` reads the environment
live; `mlx_qmm_mma` keeps a win table that can be dropped and restored).

    uv run python benchmarks/spec_exact.py --tokens 256 --prompts 5
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--prompts", type=int, default=5)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    import spec_decode as S

    from chad import mlx_qmm_mma
    from chad.cli import _pick_model
    from chad.engine import Engine

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eng = Engine(model_id=_pick_model()[0], cache_dir=None)
    eng.load()
    eng.temp, eng.top_p, eng.top_k = 0.0, 0.0, 0
    drafter = eng._dflash
    wins = dict(getattr(mlx_qmm_mma, "_WINS", {}))
    print(f"drafter={'yes' if drafter else 'NO'} kv_bits={eng.kv_bits} "
          f"qmm_mma_shapes={len(wins)}", flush=True)

    corpus = S.seed_corpus(eng.tok, root, a.prompts)

    def arm(name):
        """Set up one arm; returns a teardown."""
        os.environ.pop("CHAD_NO_QSDPA_WIDE", None)
        mlx_qmm_mma.set_wins(wins)
        eng._dflash = drafter
        if name == "serial":
            eng._dflash = None
        elif name == "spec-no-mma":
            mlx_qmm_mma.disable()
        elif name == "spec-no-qsdpa-wide":
            os.environ["CHAD_NO_QSDPA_WIDE"] = "1"

    results = {}
    for name in ("serial", "spec", "spec-no-mma", "spec-no-qsdpa-wide"):
        arm(name)
        ids = {}
        for label, messages in corpus:
            _, stats, gen_ids, _ = S.run_one(eng, eng.tok, messages, a.tokens,
                                             reset=True, tools=False)
            ids[label] = gen_ids
            print(f"  {name:20s} {label:8s} {len(gen_ids):4d} tok  "
                  f"{stats.generated_tokens / stats.gen_s:5.1f} tok/s", flush=True)
        results[name] = ids
        if a.json:
            # Save after every ARM, not at the end: this run has been stopped
            # mid-flight twice, and a completed arm is a usable result.
            import json
            with open(a.json, "w") as f:
                json.dump(results, f)
    arm("spec")   # leave the engine in its shipped configuration

    ref = results["serial"]
    w = 78
    print("\n" + "=" * w)
    print("greedy token-identity vs the serial reference")
    print("=" * w)
    print(f"{'arm':24s}{'identical':>12s}   first divergence per prompt")
    print("-" * w)
    for name, ids in results.items():
        if name == "serial":
            continue
        same, notes = 0, []
        for label, r in ref.items():
            g = ids.get(label, [])
            n = min(len(r), len(g))
            if r[:n] == g[:n]:
                same += 1
            else:
                notes.append(f"{label}@{next(i for i in range(n) if r[i] != g[i])}")
        print(f"{name:24s}{same:>7d}/{len(ref):<4d}   {', '.join(notes) or '-'}")
    print("-" * w + "\n")

    if a.json:
        import json
        with open(a.json, "w") as f:
            json.dump(results, f)
        print(f"raw ids -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
