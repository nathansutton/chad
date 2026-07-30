# GEMV / fused-kernel spike — findings (2026-07-30)

Context: [ml-explore/mlx#3313](https://github.com/ml-explore/mlx/issues/3313)
(fused transformer-layer kernels for seq_len=1 decode) was closed wontfix as
impractical for a *generic* framework. chad has one architecture (Ornith 35B
hybrid MoE) and one use case, and `mx.fast.metal_kernel` JIT-compiles custom
Metal from Python at runtime — shippable from the pure-Python uvx wheel with
no build step (precedent: `src/chad/mlx_qsdpa.py`, already shipped).

All measurements: M4 Pro 24 GB (273 GB/s), MLX 0.32.0, synthetic weights at
the exact shipped-checkpoint shapes/bits (no model load — GPU-panic rule).
Measurement pitfalls that invalidated round 1: SLC caching of repeated weights
(8-bit qmv "measured" 140% of DRAM peak) and DVFS clock ramp (stock 2-bit
"improved" 1.4x within one script). All conclusions below are from the
cache-busted, clock-pinned, interleaved harness (`spike_qmv_tune2.timed`).

## Per-token roofline (shipped 35B: UD-Q2_K_XL, 8k ctx)

~2.1 GB streamed per decode token (experts 315 MB @2-bit, GDN projections
821 MB @6-bit, lm_head 413 MB @6-bit × 248,320 vocab, attn 153 MB, shared+
router 124 MB, KV/state ~300 MB) → ideal ~130 tok/s vs 69.6 measured.

## What the spike established

1. **mlx's 6-bit qmv is near its practical limit** — ~200 GB/s (73%) at model
   shapes, better at lm_head size. Not the target.
2. **2-bit qmv is a hard plateau at ~137 GB/s / ~440 Gelem/s** (50% of peak),
   bounded by per-element instruction work, not bandwidth. Three custom
   attempts to beat it all lost cleanly:
   - schedule retune of mlx's own kernel (packs/thread, rows/simdgroup,
     simdgroups/tg): every variant ≤ stock (`spike_qmv_tune2.py`);
   - byte→float4 LUT dequant in threadgroup memory: 84 GB/s — random-index
     bank conflicts (`spike_qmv_lut.py`);
   - fp16 qdot math: 130 vs 137 GB/s — M4 does not reward halving precision
     here (`spike_expert_honest.py`).
   Corollary: the #3313 mega-kernel projection ("recover 45% of bandwidth")
   does NOT transfer to 2-bit weights — dequant rate, not dispatch overhead,
   is the binding constraint for the expert bytes. Also: 2-bit is *faster per
   element* than 4-bit (440 vs 351 Gelem/s), so the checkpoint choice stands.
3. **The MoE expert cluster is the real inefficiency**: the compiled stock
   path (gather_qmm gate|up → silu·mul → gather_qmm down → weighted sum) runs
   at 72–94 GB/s — well below the 137 plateau — because each expert is a tiny
   1024×2048 / 2048×512 GEMV and the cluster is 4+ dependent dispatches.
4. **Fused expert kernels work**: two custom dispatches built on mlx's exact
   qmv inner loop (`spike_fused_expert2.py`):
   - A: all 8 experts' gate|up in one dispatch, (j, j+512) row pairing,
     silu(g)·u in-register → 128 GB/s (93% of the 2-bit plateau);
   - B: down-projection looping experts in-register with scores folded in,
     output written once, no atomics → 90 GB/s (short K=512 rows cap ILP);
   - together: **1.25x on the cluster, ≈0.67 ms/token** (≈+5% decode),
     numerics at 9.8e-3 rel = bf16 accumulation class (same acceptance class
     as the shipped fastpath/qsdpa changes, pending greedy-parity gate).
5. Serialization tax measured: ~4.3 µs/dispatch gap, ~11–13 µs marginal cost
   for a dependent GEMV vs an overlappable one. The fused path also removes
   3 dispatch boundaries per layer — not counted in the 0.67 ms.

## Validation campaign (same day): every step of the "10–14%" path chased

Each claimed step was built and measured (honest harness throughout):

1. **Shared expert folded into A+/B+** ✓ — 6-bit branch (mlx's byte-split
   qdot, verbatim) rides in the same two dispatches; sigmoid(seg) and the
   residual add folded into B+. Isolated post-router subgraph: **1.38x,
   1.22 ms/token** (`spike_fused_expert3.py`). Numerics ~9e-3 rel (bf16
   class), routing bit-identical.
2. **router+seg concat** ✓ — 0.25 ms/token serialized; zero-padding 257→264
   rows to stay on mlx's fast-qmv path is worth 1.5 µs/step
   (`spike_concats.py`).
3. **attn q|k|v concat** ✓ but marginal — **0.03 ms/token** (3 qmv already
   overlap well). Take it, don't count on it.
4. **Kernel B 90→137 GB/s closure** ✗ REFUTED — B is flat at ~105 GB/s
   across dual-accumulator, uint32 loads, RPS 8/16, NSG 2/4/8. RPS 8→16
   changing nothing rules out h-reload cost. Root cause: down-projection
   rows are K=512 = **128 bytes** — short scattered rows stream poorly at
   any schedule. A transpose repack would fix the layout but requires
   requantization (not bit-exact) — rejected.
5. **Kernel A** — already at its plateau (48.4 µs; RPS/NSG variants lose).
6. **Router algebra swap** ✓ — `softmax(take(logits, top8))` ==
   `take(softmax(logits), top8)/sum` (monotonicity); **0/2000** selection
   mismatches on random logits; deletes the 256-wide softmax + div.
7. **Integrated compiled block** (`spike_block_chain.py` + final):
   `mx.compile` accepts the custom kernels (shippability fact — the
   integration must compile the block; uncompiled, Python graph-build costs
   ~18 µs/step and eats half the win): stock 108.1 → fused 84.4 µs/step =
   **1.28x, 0.98 ms/token including all of the above**.

## Honest bottom line

**Validated path lands ≈ +7.3% decode** (14.4 → 13.42 ms/token, 69.6 →
~74.5 tok/s projected) — NOT the 10–14% first estimated. The gap: step 4's
premise was wrong (B has no recoverable headroom), and step 3 measured 10x
smaller than guessed. The estimate died in validation; the fused-block win
itself is solid and integration-shaped (compiled, parity-classed, 2 custom
dispatches + 1 concat qmv replacing ~10 stock dispatches per layer).

Mapped but unturned (diminishing returns, parity risk): in-kernel top-k
inside A+ (~0.2–0.4 ms est., tie-order parity risk vs argpartition);
rms_norm folded into the router qmv (~0.15 ms est., another custom kernel).
6-bit GEMV is also instruction-bound (245 Gelem/s) with no scheme found —
same LUT/bank-conflict fate as 2-bit.

Gates before adoption (repo norms): greedy-token parity vs stock on real
weights, end-to-end one-load A/B on the 35B at 8k/32k, PPL check (the 33-task
set is saturated and can't authorize adoption).

## Shipped (same day): src/chad/mlx_moe_fused.py

Integration surprises vs the spike:
- The live checkpoint revision is the **q2_down3 recipe** — expert down_proj
  is 3-bit on all 40 layers (the spike measured an older all-2-bit revision
  from the same HF cache dir; `snapshots | head -1` != `refs/main`). Kernel
  B's routed section ships mlx's bits==3 qdot instead.
- The router algebra swap was REVERTED for ship: measured perf-neutral
  (84.4 vs 85.2 us, within noise) and it perturbed scores; shipped routing is
  bit-identical to stock.

Real-checkpoint gate (`gate_real35b_v2.py`, one load, in-process A/B by
flipping `layer._moe_fast`):
- decode tok/s: **+6.0% short ctx, +5.1% @8k** (second run: +7.3%/+6.7%;
  bracketed body runs agree to 0.1 tok/s within each run — call it +5-7%).
- decode-path NLL (teacher-forced S=1 over README text, the path the kernels
  live on): body 2.51026 vs fused 2.50579, **dPPL -0.45%** — fused slightly
  better (fp32 expert combine vs stock's bf16 sums). Quality gate: pass.
- greedy parity: 1/3 prompts identical over 200 tokens; 2/3 diverge (tokens
  92 and 4) with equal-or-better NLL — accumulation-order rounding, the
  known acceptance class; bit-identity is unreachable in principle since the
  fused combine is *more* precise than stock.
- peak memory unchanged (15.7 GB); transforms replace originals, no copies.
