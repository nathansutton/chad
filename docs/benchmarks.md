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

| Model | Prefill (cold) | Decode | Warm-step prefill |
|---|---|---|---|
| **Qwen3.8-27B** `UD-Q3_K_XL-MTP` (shipped) | *run `chad-bench`* | *run `chad-bench`* | ~0.2 s (16 tok) |

> **2.0.0 changed the default model**, and the prefill/decode columns above are not yet
> re-measured on it — rather than carry Ornith's numbers under a new model's name, they are
> left for `chad-bench` to fill in on your own machine. The retired Ornith figures are kept
> below under [Historical: Ornith](#historical-ornith-35b--9b).

Two things hold whatever the model. Prefill rate **falls as context grows** (attention is
quadratic), so a cold-prompt number measured at 5,000 tokens is materially lower by 32k.
Decode is the opposite: nearly flat, and it is what the long-context tuning defends. The
warm-step number is the one that decides how the agent *feels*, and it is a property of the
persistent prefix cache rather than of the weights — a follow-up turn prefills the ~16
tokens it appended, not the whole transcript.

Qwen3.8-27B is dense, so unlike the retired 35B MoE (which activated only ~3B params per
token) every parameter is read on every token. That is why the quant is aggressive: on a
dense model, shrinking the weights is the only decode lever there is.

## The agentic-loop win: ~0.2 s per step, not ~10 s

The headline isn't the cold-prefill rate — it's what a *follow-up* turn costs. Taking the
retired 35B's measured numbers as the worked example, a 5,000-token transcript prefills
cold in ~6.8 s. But the next agentic step only appends
the model's reply, a tool call, and the tool's output, so with the persistent prefix cache
it re-reads **nothing**: the follow-up turn prefills just the ~16 appended tokens in
**~0.2 s**.

```
cache-less backend:  re-prefill all 5,144 tokens  ->  ~6.8 s of dead air, every step
chad (prefix cache): prefill the 16 new tokens     ->  ~0.2 s, every step
```

That ~30× gap is the entire reason a local model can feel like an agent instead of a batch
job — and it widens with the transcript, since the cache-less side grows while the warm
step stays flat. The ratio is a property of the cache, not of the weights, so it survives the
model change. Why the cache is *append-only* (and why that's the right trade for a hybrid
SSM/attention model) is in
[the cache trade](design.md#trimmable-vs-append-only-the-cache-trade-chad-lives-with).

## Why decode sits where it does

The intuitive answer is "memory bandwidth" — each token streams the resident weights through
the chip once, so `tok/s ≈ bandwidth / resident-bytes-per-token`. On a **dense** model like
the shipped 27B that first-order story is largely right, and it is the reason the quant is
as aggressive as it is: every parameter is read for every token, so shrinking the weights is
the only decode lever there is. ~12 GB of weights against this M4 Pro's ~273 GB/s is the
envelope the engine works inside.

> The analysis below was measured on the **retired Ornith 35B MoE**. It is kept because the
> lessons generalize — but a sparse MoE and a dense hybrid are limited by different things,
> and 2.0.0 deleted the MoE-specific kernels along with the model, since they could never
> install on a dense checkpoint.

Bandwidth was **not** what limited the 35B MoE, and that mattered for anyone trying to make
it faster. In-situ ablation of its decode step put bandwidth-minimal cost around 9 ms against
~14 ms real: the step issued roughly **400 Metal kernels per token**, each carrying ~9 µs of
launch and gap latency, so it was **dispatch-bound, not bandwidth-bound**. Two consequences:

- Work that removes *kernels* pays; work that removes *bytes* often doesn't. chad's decode
  fast-path ([`mlx_fastpath.py`](../src/chad/mlx_fastpath.py)) fuses expert and GDN
  projections and compiles the whole S=1 layer step, removing ~150 dispatches per token.
  On top of that, the 35B's per-token MoE block ran as two custom Metal kernels — one for
  all routed + shared gate|up projections, one folding the down-projections, routing
  weights, and residual add — worth another 5–7% decode (70.7 → 74.3 tok/s at 8k context),
  with routing bit-identical to stock. That code is gone as of 2.0.0; the dense fastpath
  it sat beside is what ships.
- Attention is the exception, and it's a reuse problem rather than a fetch problem. Ablating
  the math out of the fused quantized-KV kernel leaves it streaming at ~331 GB/s — already at
  this machine's measured roofline — with ~68% of its runtime spent on the 8 GQA q-heads
  re-reading staged K/V out of threadgroup memory. That's why
  [`mlx_qsdpa.py`](../src/chad/mlx_qsdpa.py) has a `simdgroup_matrix` schedule that gives each
  simdgroup its own position-stream instead of its own head. This part *does* carry over: the
  same kernel serves the shipped model, and 2.0.0 added a multi-token (S>1) tier to it for
  speculative verification and prefill.

The honest caveat, and the reason the numbers above are end-to-end rather than per-kernel:
**isolated kernel speedups oversell badly.** The attention retile is a measured 1.26–1.40×
on the kernel and moves the whole decode step ~2%, because attention is only a fifth of it.
Trust steady-state `chad-bench` tok/s; don't extrapolate from a microbenchmark. 🗿

You don't have to tune any of this. chad picks the fast configuration at startup — the
KV-cache bit width, the fused-attention schedule (including its split factor, which widens
past 16k context), and the speculative draft depth are chosen from measurements and applied
for you. MLX's own runtime knobs are deliberately left alone: `MLX_METAL_FAST_SYNCH` and the
command-buffer size limits were swept and every setting was *slower* than mlx's defaults, so
chad overrides none of them. `chad-bench` reports what you're getting.

## The model: Qwen3.8-27B

chad runs **one** model — [`Qwen3.8-27B UD-Q3_K_XL-MTP`](https://huggingface.co/nathansutton/Qwen3.8-27B-UD-Q3_K_XL-MTP-MLX),
a dense `qwen3_5` hybrid (64 layers: 48 GatedDeltaNet + 16 full attention) quantized to
3-bit group-64 with `lm_head` held at 5-bit, ~12 GB resident. There is no RAM-aware pick and
no size tier: 2.0.0 retired the Ornith 35B/9B pair, and chad targets 24 GB Macs and nothing
smaller. The vision tower is not present at all (it was dropped at conversion time; mlx-lm's
`qwen3_5` loader would have discarded it at load anyway).

It's a *thinking* model that emits tool calls in the XML `<function=…>` dialect — the
harness parses both that and JSON, and strips `<think>` blocks. It also ships its trained
**multi-token-prediction head**, which is what the engine drafts with; see
[speculative decoding](configuration.md#speculative-decoding--kernel-knobs). One fast, good
model per machine beats a menu of mediocre ones, so there are no model flags to pick from —
you just run `chad`.

## Two throughput levers

**Thinking budget.** This is a reasoning model, and its `<think>` blocks dominate what it
generates — measured on real agentic traces, **~62–66% of all generated tokens**. Because decode runs at a roughly fixed tok/s whatever it is writing,
those tokens are pure wall-clock. `--no-think` (a per-turn toggle, or the flag) injects an empty think block
and skips them — the most effective time-to-done lever on well-scoped agentic work. Thinking
stays **on** by default, since it helps on harder reasoning; flip it off when the task is
well-specified and you'd rather not wait on the reasoning tokens.

**Speculative decoding.** The shipped checkpoint carries its own trained multi-token-prediction
head, so chad drafts with it and verifies in one batched forward, accepting by exact rejection
sampling — greedy output stays token-identical, sampled output keeps the model's true
distribution. Measured on an M4 Pro at temp 1.0: **1.38× on quote-heavy spans, 1.11× on novel
code, 1.0× on free prose.** It speeds up predictable text and costs nothing on the rest.

Prompt-lookup decoding (PLD) — the draft-model-free variant that proposes continuations from
n-grams already in context — is implemented and greedy-exact, but is **opt-in** (`CHAD_USE_PLD=1`)
rather than default. It drafts from context *recurrence*, and on real agentic traces that is a
minority of what this agent writes: whole-session contribution measured at **+2.2%** of
generated tokens. It also doesn't compose with the MTP path (one generate loop each), so on a
self-speculating checkpoint the head is strictly the better of the two.

---

## Task pass-rates: Terminal-Bench 2.1 (reproducible)

Capability is measured on [Terminal-Bench 2.1](https://www.tbench.ai/leaderboard), and the
whole run is reproducible **from a Mac**: the exact Harbor agent adapter, the runner script,
and the serving recipe live in [`benchmarks/tb2/`](../benchmarks/tb2/README.md). Serve a
model with `mlx_lm.server` (or a GGUF from a GPU box, if yours can't hold the weights), point
the kit at it, and check the number yourself.

**The 2.0.0 default has not been scored yet.** Changing the default model invalidates the
score measured on the old one, and chad's release rule is that a benchmark claim flips in its
own dedicated commit once the run lands — never bundled into the release that changed the
thing being measured. So 2.0.0 ships with no pass-rate claim.

Mind the kit's standing caveat when you run it: TB2 pass-rate is **throughput-sensitive**.
Tasks carry wall-clock budgets, so your serving speed is part of the score, and a slow host
can fail a task on time rather than on capability.

### Historical: Ornith 35B / 9B

Kept for the record, and because the reproduction kit still works against these weights.
**None of this describes what you get from chad 2.0.0** — the Ornith models were retired as
defaults in that release.

<img src="tbench-size-vs-score.png" width="840" alt="Terminal-Bench 2.1: accuracy vs. cost per run. Every verified entry is a proprietary frontier model in a datacenter, costing $130–$2,000 per run. chad + Ornith (a 35B MoE) clears 57% on an Apple Silicon laptop for the electricity — the only no-API-cost point on the board.">

| Measured | Model | Score | Conditions |
|---|---|---|---|
| chad 1.x, 2026 | Ornith-1.0-35B `UD-Q2_K_XL` | **57%** (51/89) | k=1, self-run, **not leaderboard-verified** |

The comparison the chart makes was the point of it: every *verified* entry on the TB 2.1
board is a proprietary frontier model in a datacenter, scoring 59–84% and spending
$130–$2,000 in API fees per run. That result sat at 57% for the electricity. The axis is
capability per dollar, and it is the reason chad exists — but the number itself belongs to a
model chad no longer ships, and is recorded here rather than claimed on the front page.

Throughput for the retired pair, measured with `chad-bench` on an M4 Pro (5,000-token cold
prompt, 128-token decode, a follow-up turn appending ~16 tokens):

| Model | Macs | Prefill (cold) | Decode | Warm-step prefill |
|---|---|---|---|---|
| Ornith-1.0-35B (2-bit MoE) | ≥ 24 GB | ~730 tok/s | ~74 tok/s | ~0.2 s (16 tok) |
| Ornith-1.0-9B (4-bit dense) | 16 / 18 GB | ~360 tok/s | ~46 tok/s | ~0.25 s (16 tok) |

The 35B MoE activated only ~3B params/token, which is why it decoded ~1.6× faster than the
9B dense model despite 4× the parameters.

---

*Day-to-day correctness is additionally tracked in a private eval suite (it seeds repos,
runs the agent, and verifies the actual edit). This page stays focused on the numbers you
can reproduce yourself: throughput via `chad-bench`, pass-rates via `benchmarks/tb2/`.*
