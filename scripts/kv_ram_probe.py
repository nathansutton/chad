#!/usr/bin/env python3
"""Plan 036 — measure the live KV-cache memory cost per token for the default model.

Loads the engine as chad does, prefills increasing context, and reports MLX active
memory + per-token KV bytes (attention KVCache layers grow; SSM/DeltaNet ArraysCache is a
fixed floor) against the Metal recommended working-set budget. Grounds the RAM-aware
compaction trigger (no magic 120k). One-time measurement; loads the model.
"""
import os
import sys

import mlx.core as mx
from mlx_lm.models import cache as cache_utils

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from chad.cli import _pick_model  # noqa: E402
from chad.engine import Engine  # noqa: E402

GB = 1e9


def kv_nbytes(cache):
    total = 0
    for c in cache:
        if isinstance(c, cache_utils.KVCache):
            for arr in (getattr(c, "keys", None), getattr(c, "values", None)):
                if arr is not None:
                    total += arr.nbytes
    return total


def main():
    info = mx.device_info()
    budget = info["max_recommended_working_set_size"]
    print(f"device: {info['device_name']}  mem={info['memory_size']/GB:.1f} GB  "
          f"recommended_working_set={budget/GB:.2f} GB")

    model_id, _why = _pick_model()
    print(f"model: {model_id}")
    eng = Engine(model_id=model_id, draft_id=None, prompt_lookup=False)
    eng.load()
    mx.clear_cache()
    n_kv = sum(1 for c in eng._cache if isinstance(c, cache_utils.KVCache))
    n_ssm = sum(1 for c in eng._cache if isinstance(c, cache_utils.ArraysCache))
    print(f"layers: {n_kv} attention (grow) + {n_ssm} SSM (fixed floor)")
    print(f"effective_ctx={eng.effective_ctx}  active_mem after load={mx.get_active_memory()/GB:.2f} GB")

    pts = []
    for n in (2000, 22000, 42000):
        eng._reset_cache()
        mx.clear_cache()
        ids = list(range(100, 100 + n))
        eng._prefill(ids)
        mx.eval([c.state for c in eng._cache])
        mx.clear_cache()  # drop transient prefill scratch; keep the persistent KV cache
        kv = kv_nbytes(eng._cache)
        active = mx.get_active_memory()
        pts.append((n, kv, active))
        print(f"  ctx={n:>6}: KV={kv/GB:.3f} GB  active_mem={active/GB:.2f} GB  "
              f"headroom_to_budget={(budget-active)/GB:.2f} GB")

    (n0, kv0, _), (n1, kv1, _) = pts[0], pts[-1]
    bpt = (kv1 - kv0) / (n1 - n0)
    print(f"\nKV bytes/token = {bpt:,.0f}  ({bpt/1024:.2f} KiB/tok)")
    # active_mem bytes/token (includes any per-token overhead beyond raw KV arrays)
    ampt = (pts[-1][2] - pts[0][2]) / (n1 - n0)
    print(f"active_mem bytes/token = {ampt:,.0f}  ({ampt/1024:.2f} KiB/tok)")
    base_active = pts[0][2] - ampt * n0  # extrapolated model+floor at ctx=0
    print(f"base active (model+SSM floor, ctx->0) ~= {base_active/GB:.2f} GB")
    for frac in (0.85, 0.90):
        usable = budget * frac - base_active
        print(f"  at {int(frac*100)}% of budget: ~{int(usable/ampt):,} tokens of context "
              f"before compaction")


if __name__ == "__main__":
    main()
