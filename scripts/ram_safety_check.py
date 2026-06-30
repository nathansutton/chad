#!/usr/bin/env python3
"""Plan 036 Lever A — OOM-safety validation for the RAM-aware compaction trigger.

Peak memory is linear in context: peak(ctx) = base + slope*ctx (KV grows per-token; the
prefill/decode scratch is bounded). So we measure peak at two real context sizes (each
includes a short generation on top = decode scratch peak), fit the line, and extrapolate
to the actual ctx_limit cli.py would pick. Asserts the extrapolated peak stays under the
Metal recommended working set — the plan-036 STOP condition. Faster than prefilling the
full window; loads the model. Output is unbuffered + flushed (survives a timeout kill).
"""
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from chad.cli import _pick_model, ram_aware_ctx_limit  # noqa: E402
from chad.engine import Engine  # noqa: E402

GB = 1e9


def p(*a):
    print(*a, flush=True)


def peak_at(eng, ctx):
    """Prefill to `ctx` tokens, generate a little on top, return peak bytes."""
    mx.reset_peak_memory()
    eng._reset_cache()
    ids = list(range(100, 100 + ctx))
    eng._prefill(ids)
    mx.eval([c.state for c in eng._cache])
    eng._cached_ids = list(ids)
    _t, st = eng.generate(ids + [200, 201], max_tokens=40)
    peak = mx.get_peak_memory()
    p(f"  ctx={ctx:>7,}: active={mx.get_active_memory()/GB:.2f} GB  peak={peak/GB:.2f} GB  "
      f"({st.generated_tokens} tok @ {st.tok_per_s:.0f} tok/s)")
    return peak


def main():
    info = mx.device_info()
    budget = info["max_recommended_working_set_size"]
    model_id, _ = _pick_model()
    eng = Engine(model_id=model_id, draft_id=None, prompt_lookup=False)
    eng.load()
    mx.clear_cache()
    active = mx.get_active_memory()
    ctx_limit = ram_aware_ctx_limit(eng.effective_ctx, budget, active,
                                    eng.kv_bytes_per_token)
    p(f"budget(working_set)={budget/GB:.2f} GB  active_after_load={active/GB:.2f} GB")
    p(f"kv_bytes_per_token={eng.kv_bytes_per_token:,.0f}  ctx_limit={ctx_limit:,}  "
      f"window={eng.effective_ctx:,}")

    c0, c1 = 50_000, 100_000
    p("measuring peak vs context (each + a 40-tok generation):")
    y0 = peak_at(eng, c0)
    y1 = peak_at(eng, c1)
    slope = (y1 - y0) / (c1 - c0)
    base = y0 - slope * c0
    # Extrapolate to the trigger, and to trigger + a full max generation on top (the true
    # worst case: prompt at the threshold, then a turn decodes max_gen more before the
    # NEXT step's compaction check fires).
    extrap = base + slope * ctx_limit
    worst = base + slope * (ctx_limit + 8192)
    p(f"\nfit: peak ≈ {base/GB:.2f} GB + {slope*1e6:.3f} MB/1k-tok")
    p(f"extrapolated peak @ ctx_limit({ctx_limit:,}) = {extrap/GB:.2f} GB  "
      f"({extrap/budget*100:.1f}% of budget)")
    p(f"worst case @ ctx_limit+8k gen     = {worst/GB:.2f} GB  "
      f"({worst/budget*100:.1f}% of budget)")
    p(f"headroom at worst case = {(budget-worst)/GB:.2f} GB to working-set, "
      f"{(info['memory_size']-worst)/GB:.2f} GB to physical")
    ok = worst < budget
    p("\nPASS: worst-case peak under working-set budget" if ok
      else "\nFAIL: worst-case peak EXCEEDS budget — tighten reserve/safety (036 STOP)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
