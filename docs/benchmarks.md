# Throughput & performance

*The numbers that decide whether a local agent feels responsive — prefill speed, decode
speed, and the per-step cost once the cache is warm. All of them are reproducible on your
own Mac with `chad-bench`. Task pass-rates (Terminal-Bench 2.1) are reproducible too —
they have their own kit in [`benchmarks/tb2/`](../benchmarks/tb2/README.md). For the
engineering behind them, see [Design & internals](design.md).*

## Reproduce it yourself

```bash
uv run chad-bench                       # default: 5000-token prefill, 128-token decode
uv run chad-bench --prefill-tokens 8000 --gen-tokens 256
```

It drives the **real** `Engine` on the **real** model (`src/chad/bench.py`) and reports
three things:

1. **Prefill (cold)** — how fast the model reads a fresh prompt. The bill a naive agentic
   loop pays *every step*.
2. **Decode** — how fast it writes new tokens. Memory-bandwidth bound, roughly constant.
3. **Warm step** — the agentic-loop number: how few tokens a *follow-up* turn has to
   prefill once the persistent prefix cache is warm. This is the whole point.

## Measured throughput (M4 Pro, 24 GB)

Measured with `chad-bench` (5,000-token cold prompt, 128-token decode, a follow-up turn
that appends ~16 tokens):

| Model | Macs | Prefill (cold) | Decode | Warm-step prefill |
|---|---|---|---|---|
| **Ornith-1.0-35B** (2-bit MoE, default) | ≥ 24 GB | **~730 tok/s** | **~74 tok/s** | ~0.2 s (16 tok) |
| **Ornith-1.0-9B** (4-bit dense, fallback) | 16 / 18 GB | **~360 tok/s** | **~46 tok/s** | ~0.25 s (16 tok) |

The 35B MoE activates only ~3B params/token, so despite 4× the parameters it decodes
**~1.6× faster** than the 9B dense model — and the persistent prefix cache keeps it there.

Prefill rate falls as context grows (attention is quadratic), so the 35B's ~730 tok/s at
a 5,000-token prompt is ~500 tok/s by 32k. Decode is the opposite: nearly flat, and what
the long-context tuning above defends.

## The agentic-loop win: ~0.2 s per step, not ~10 s

The headline isn't the cold-prefill rate — it's what a *follow-up* turn costs. On the 35B,
a 5,000-token transcript prefills cold in ~6.8 s. But the next agentic step only appends
the model's reply, a tool call, and the tool's output, so with the persistent prefix cache
it re-reads **nothing**: the follow-up turn prefills just the ~16 appended tokens in
**~0.2 s**.

```
cache-less backend:  re-prefill all 5,144 tokens  ->  ~6.8 s of dead air, every step
chad (prefix cache): prefill the 16 new tokens     ->  ~0.2 s, every step
```

That ~30× gap is the entire reason a local model can feel like an agent instead of a batch
job — and it widens with the transcript, since the cache-less side grows while the warm
step stays flat. Why the cache is *append-only* (and why that's the right trade for Ornith) is in
[the cache trade](design.md#trimmable-vs-append-only-the-cache-trade-chad-lives-with).

## Why decode sits where it does

The intuitive answer is "memory bandwidth" — each token streams the resident weights
through the chip once, so `tok/s ≈ bandwidth / resident-bytes-per-token`. That's the right
first-order story for the 9B dense model, whose ~5 GB of 4-bit weights cap decode near
**~46 tok/s** on this M4 Pro (~273 GB/s).

It is **not** what limits the 35B MoE, and that matters if you're thinking about making it
faster. In-situ ablation of its decode step puts bandwidth-minimal cost around 9 ms against
~14 ms real: the step issues roughly **400 Metal kernels per token**, each carrying ~9 µs of
launch and gap latency, so the 35B is **dispatch-bound, not bandwidth-bound**. Two
consequences:

- Work that removes *kernels* pays; work that removes *bytes* often doesn't. chad's decode
  fast-path ([`mlx_fastpath.py`](../src/chad/mlx_fastpath.py)) fuses expert and GDN
  projections and compiles the whole S=1 layer step, removing ~150 dispatches per token.
- Attention is the exception, and it's a reuse problem rather than a fetch problem. Ablating
  the math out of the fused quantized-KV kernel leaves it streaming at ~331 GB/s — already at
  this machine's measured roofline — with ~68% of its runtime spent on the 8 GQA q-heads
  re-reading staged K/V out of threadgroup memory. That's why
  [`mlx_qsdpa.py`](../src/chad/mlx_qsdpa.py) has a `simdgroup_matrix` schedule that gives each
  simdgroup its own position-stream instead of its own head.

The honest caveat, and the reason the numbers above are end-to-end rather than per-kernel:
**isolated kernel speedups oversell badly.** The attention retile is a measured 1.26–1.40×
on the kernel and moves the whole decode step ~2%, because attention is only a fifth of it.
Trust steady-state `chad-bench` tok/s; don't extrapolate from a microbenchmark. 🗿

You don't have to tune any of this. chad picks the fast configuration at startup — the
KV-cache bit width and the fused-attention schedule (including its split factor, which
widens past 16k context) are chosen from measurements on this model and applied for you.
MLX's own runtime knobs are deliberately left alone: `MLX_METAL_FAST_SYNCH` and the
command-buffer size limits were swept on the 35B and every setting was *slower* than
mlx's defaults, so chad overrides none of them. `chad-bench` reports what you're getting.

## The model: Ornith

chad runs **one** model from the Ornith family, picked by your RAM: the **35B** MoE
(2-bit experts) on ≥24 GB Macs, the **9B** dense (4-bit) on 16/18 GB ones (see
[Quickstart](../README.md#quickstart)). Both are converted to MLX with a mixed-precision
quant (the vision tower is stripped at load by mlx-lm's `qwen3_5` loader). It's a *thinking*
model that emits tool calls in the XML `<function=…>` dialect — the harness parses both that
and JSON, and strips `<think>` blocks. One fast, good model per machine beats a menu of
mediocre ones, so there are no model flags to pick from — you just run `chad`.

## Two throughput levers

**Thinking budget.** Ornith is a reasoning model, and its `<think>` blocks run **36–41% of
all generated tokens**. Because decode runs at a roughly fixed tok/s whatever it is writing,
those tokens are pure wall-clock. `--no-think` (a per-turn toggle, or the flag) injects an empty think block
and skips them — the most effective time-to-done lever on well-scoped agentic work. Thinking
stays **on** by default, since it helps on harder reasoning; flip it off when the task is
well-specified and you'd rather not wait on the reasoning tokens.

**Prompt-lookup decoding (PLD).** Speculative decoding without a draft model — when the
model is about to re-quote text already in context (a file it just `read` into an `edit`),
an n-gram lookup proposes the continuation and one batched forward accepts a whole run,
exact at temperature 0. It's implemented and provably greedy-identical, **but it requires a
trimmable KV cache and Ornith doesn't have one**, so it's gated off for the shipped model
and falls back cleanly. It helps trimmable research models (e.g. Qwen2.5-Coder) on
edit-heavy work. The full reason is in
[the cache trade](design.md#trimmable-vs-append-only-the-cache-trade-chad-lives-with).

---

## Task pass-rates: Terminal-Bench 2.1 (reproducible)

Capability is measured on [Terminal-Bench 2.1](https://www.tbench.ai/leaderboard), the
public benchmark behind the chart in the [README](../README.md) — and the whole run is
reproducible **from a Mac**: the exact Harbor agent adapter, the runner script, and the
serving recipe live in [`benchmarks/tb2/`](../benchmarks/tb2/README.md). Serve the
shipped Ornith-35B MLX quant with `mlx_lm.server` on a ≥32 GB Mac (or a GGUF from a GPU
box if, like our dev machine, yours can't hold the 35B), point the kit at it, and check
the number yourself. The full verified reference run is still in flight — treat
published points as provisional until it lands, and mind the kit's caveat that TB2
pass-rate is **throughput-sensitive** (tasks have wall-clock budgets, so your serving
speed is part of the score).

---

*Day-to-day correctness is additionally tracked in a private eval suite (it seeds repos,
runs the agent, and verifies the actual edit). This page stays focused on the numbers you
can reproduce yourself: throughput via `chad-bench`, pass-rates via `benchmarks/tb2/`.*
