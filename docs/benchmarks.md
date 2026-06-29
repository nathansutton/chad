# Throughput & performance

*The numbers that decide whether a local agent feels responsive — prefill speed, decode
speed, and the per-step cost once the cache is warm. All of them are reproducible on your
own Mac with `chad-bench`; none of them are pass-rates from a suite you can't run. For the
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
| **Ornith-1.0-35B** (2-bit MoE, default) | ≥ 24 GB | **~500 tok/s** | **~71 tok/s** | ~0.2 s (16 tok) |
| **Ornith-1.0-9B** (4-bit dense, fallback) | 16 / 18 GB | **~360 tok/s** | **~46 tok/s** | ~0.2 s (16 tok) |

The 35B MoE activates only ~3B params/token, so despite 4× the parameters it decodes
**~1.5× faster** than the 9B dense model — and the persistent prefix cache keeps it there.

## The agentic-loop win: ~0.2 s per step, not ~10 s

The headline isn't the cold-prefill rate — it's what a *follow-up* turn costs. On the 35B,
a 5,000-token transcript prefills cold in ~9.9 s. But the next agentic step only appends
the model's reply, a tool call, and the tool's output, so with the persistent prefix cache
it re-reads **nothing**: the follow-up turn prefills just the ~16 appended tokens in
**~0.2 s**.

```
cache-less backend:  re-prefill all 5,144 tokens  ->  ~9.9 s of dead air, every step
chad (prefix cache): prefill the 16 new tokens     ->  ~0.2 s, every step
```

That ~50× gap is the entire reason a local model can feel like an agent instead of a batch
job. Why the cache is *append-only* (and why that's the right trade for Ornith) is in
[the cache trade](design.md#trimmable-vs-append-only-the-cache-trade-chad-lives-with).

## Why decode sits where it does

Token generation is **memory-bandwidth bound**: each token streams the resident weights
through the chip once, so `tok/s ≈ bandwidth / resident-bytes-per-token`. On this M4 Pro
(~273 GB/s) the 9B's ~5 GB of 4-bit weights cap decode near **~46 tok/s**; the 35B MoE
touches fewer bytes per token (sparse experts) and lands near **~71 tok/s**. This is a
property of the machine, not the harness — it's the ceiling Chad runs *at*, not past. "As
fast as a MacBook Pro gets — no faster." 🗿

## The model: Ornith

Chad runs **one** model from the Ornith family, picked by your RAM: the **35B** MoE
(2-bit experts) on ≥24 GB Macs, the **9B** dense (4-bit) on 16/18 GB ones (see
[Quickstart](../README.md#quickstart)). Both are converted to MLX with a mixed-precision
quant (the vision tower is stripped at load by mlx-lm's `qwen3_5` loader). It's a *thinking*
model that emits tool calls in the XML `<function=…>` dialect — the harness parses both that
and JSON, and strips `<think>` blocks. One fast, good model per machine beats a menu of
mediocre ones, so there are no model flags to pick from — you just run `chad`.

## Two throughput levers

**Thinking budget.** Ornith is a reasoning model, and its `<think>` blocks run **36–41% of
all generated tokens**. Because decode is bandwidth-bound at a fixed tok/s, those tokens are
pure wall-clock. `--no-think` (a per-turn toggle, or the flag) injects an empty think block
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

*Correctness and task-completion quality are tracked separately in a private eval suite
(it seeds repos, runs the agent, and verifies the actual edit). This page stays focused on
the throughput numbers you can reproduce yourself.*
