# Throughput & performance

*Three numbers decide whether a local agent feels responsive: prefill speed, decode speed,
and the per-step cost once the cache is warm. This page has all three, plus what chad's
engine buys over a stock engine on the same weights. Every number is a hardware measurement
you can reproduce on your own Mac (`chad-bench` for chad, `benchmarks/stock/` for the
comparison). Nothing here is a score against anyone else: chad's premise is a 24 GB laptop,
and the only comparisons that honour it are between engines on that laptop. For the
engineering behind the numbers see [Design & internals](design.md).*

## Reproduce it yourself

```bash
uv run chad-bench                       # default: 5000-token prefill, 128-token decode
uv run chad-bench --prefill-tokens 8000 --gen-tokens 256
uv run chad-bench --chunk 1024          # force a prefill chunk size (overrides CHAD_PREFILL_CHUNK)
uv run chad-bench --agentic             # the cache-miss benchmark, below
```

It drives the **real** `Engine` on the **real** model (`src/chad/bench.py`) and reports
three things:

1. Prefill (cold). How fast the model reads a fresh prompt. The bill a naive agentic
   loop pays *every step*.
2. Decode. How fast it writes new tokens. Memory-bandwidth bound, roughly constant.
3. Warm step. The agentic-loop number: how few tokens a *follow-up* turn has to prefill
   once the persistent prefix cache is warm. This is the whole point.

## Measured throughput (M4 Pro, 24 GB)

Measured with `chad-bench` (5,000-token cold prompt, 128-token decode, a follow-up turn
that appends ~16 tokens):

| Model | Prefill (cold) | Decode | Warm-step prefill |
|---|---|---|---|
| **Qwen3.8-27B** `Ternary-Bonsai-2-DFlash2` (shipped, default) | ~97 tok/s | ~60 tok/s | ~0.56 s (16 tok) |
| same, serial (`CHAD_NO_DFLASH=1`) | ~99 tok/s | ~21 tok/s | ~0.56 s (16 tok) |
| Qwen3.8-27B `UD-Q3_K_XL-DFlash2` (3-bit, `--model`) | ~104 tok/s | ~59 tok/s | ~0.55 s (16 tok) |
| same, serial (`CHAD_NO_DFLASH=1`) | ~105 tok/s | ~18 tok/s | ~0.55 s (16 tok) |

> Measured on this machine with the command above. Run it on yours; these are hardware
> numbers, not scores.

Prefill is the honest cost of a dense checkpoint: every one of the 27B parameters is read
for every token of the prompt, so ~105 tok/s is close to this chip's compute roofline rather
than a tuning failure. It is also why the [warm prefix
cache](#the-second-session-in-a-project-starts-warm) matters more than any decode work: the
third column, not the first, is what a session actually pays after its first turn.

The default-row decode figure flatters the drafter: `chad-bench` tiles a block of code, the
drafter accepts nearly all of it, and a 128-token run is short. On real mid-session agent
contexts the same engine measures ~2× serial rather than ~3×, and the
[speculative-decoding tables](#two-throughput-levers) are the numbers a session lives at.
The warm step varies 0.5-0.75 s run to run: 16 tokens of prefill plus the fixed cost of one
forward.

`--agentic` measures a different thing and is worth knowing about: it seeds a large context
(`--context-tokens`, default 24,000) and reproduces the **truncated-turn cache miss**, the
case where a step ends mid-`<think>` and the next step's prompt is no longer an extension of
the cache. It reports the re-prefill that step pays with and without the fix. The other three
numbers tell you how fast the happy path is; this one tells you what an unhappy one costs.

Two things hold whatever the model. Prefill rate **falls as context grows** (attention is
quadratic), so a cold-prompt number measured at 5,000 tokens is materially lower by 32k.
Decode is the opposite: nearly flat, and it is what the long-context tuning defends. The
warm-step number is the one that decides how the agent *feels*, and it is a property of the
persistent prefix cache rather than of the weights: a follow-up turn prefills the ~16 tokens
it appended, not the whole transcript.

Qwen3.8-27B is **dense**: every parameter is read on every token, with no sparse routing to
hide behind. That is why the quant is aggressive: on a dense model, shrinking the weights is
the only decode lever there is, which is also [why decode sits where it
does](#why-decode-sits-where-it-does).

## Same model, same Mac, stock engine

The question chad has to answer is what its engine buys over a generic local-model tool
pointed at the **same weights on the same laptop**. So: Qwen3.8-27B at the `UD-Q3_K_XL`
recipe (Unsloth's GGUF for llama.cpp, chad's MLX conversion of the same per-tensor bit map,
the 3-bit alternative chad runs with `--model`; the shipped ternary build has no GGUF to
compare against) on the same M4 Pro (24 GB), one engine resident at a time, each measured with its own native
benchmark on a 512-token prompt and a 128-token generation. It was measured once on the same 
GGUF (0.32.15, a `FROM`-only Modelfile, `num_ctx` 2048, temperature 0, timed from 
`/api/generate`'s own counters) at 96 tok/s prefill and 10.9 tok/s decode, the llama.cpp 
number, as expected. The row is kept under `_runs/ollama.json`. There is no script arm 
because importing a GGUF into Ollama needs ~45 GB of scratch disk for nothing new.

| Engine | Prefill (512-tok prompt) | Decode (128 tok) | Speculative decoding |
|---|---|---|---|
| llama.cpp `llama-bench` (stock, build 10470) | 102 tok/s | 10.9 tok/s | off in this benchmark |
| llama.cpp `llama-server` (build 10917), serial | 97 tok/s | 11.3 tok/s | off |
| llama.cpp `llama-server` (build 10917) | 95 tok/s | 11.1 tok/s | DFlash2 drafter (Q4_K_M GGUF) |
| **chad**, serial (`CHAD_NO_DFLASH=1`) | 100 tok/s | 17.9 tok/s | off |
| **chad**, default | 101 tok/s | **62.9 tok/s** | DFlash2 block drafter |

Reproduce it with `uv run python benchmarks/stock/stock.py llama`, `… llama-dflash` and
`… chad`, one arm at a time since each loads ~13 GB, then `… table` to render the rows.
The two `llama-server` rows are one build and one instrument, serial and drafted, on the
same 512-token prompt `chad-bench` tiles; `llama-dflash` needs build 10658 or later
(`STOCK_LLAMA_BIN` points at an unpacked release without touching a brew install). The measured rows are
committed under `benchmarks/stock/_runs/`. How to read it:

- Prefill is a wash. Both engines read a 512-token prompt at ~100 tok/s: a dense 27B
  reads every parameter for every prompt token, and that is the chip's compute roofline,
  not anyone's tuning.
- Serial decode is the bandwidth wall, and the two rows sit on it differently. The
  MLX quant is ~12 GB resident against the GGUF's 13.1, and chad's serial step runs the
  fused single-token kernels described [below](#why-decode-sits-where-it-does):
  concatenated `gate|up` and `in_proj` matmuls, a compiled layer step. llama.cpp's serial
  step for this hybrid GatedDeltaNet/attention architecture was not profiled; the row says
  what a fitted engine buys on this checkpoint, not what llama.cpp can do in general.
- The drafter is the gap that matters, and what decides it is the verify pass.
  llama.cpp has run DFlash2 drafters since build 10658 (`--spec-type draft-dflash`), and
  the drafter is published in GGUF form (`incoai/Qwen3.8-27B-DFlash2-GGUF`, BF16 / Q8_0 /
  Q4_K_M). On this checkpoint it works as a drafter, 96.5% of drafted tokens accepted on
  the tiled prompt, and decodes at 11.1 tok/s against 11.3 serial anyway. A round verifies
  8 tokens in one batch, and llama.cpp's Metal path runs a batch of 1 / 2 / 4 / 8 / 16 at
  11.2 / 12.8 / 13.5 / 14.0 / 45.5 tok/s (`llama-bench -p 1,2,4,8,16 -n 0`, recorded in
  `_runs/llama-dflash.json`): the 8-token verify alone costs ~6.4 serial steps, so no
  acceptance rate can pay for the round, and the jump at 16 is a kernel switch the round
  sits just below. chad's whole round at the same width, drafter included, costs ~2.2
  serial steps (`mlx_dflash.BLOCK_ROUND_COSTS`, the flat verify described
  [below](#two-throughput-levers)), which is the difference between the two drafted rows.
  llama.cpp's DFlash2 PR reports ~1.8× on an M5 Pro with a Q4_K_M target, so this measures
  this GGUF on this Mac, not llama.cpp's DFlash2 in general.
- 63 is a ceiling, and a session runs slower. `chad-bench` tiles a block of code, the
  drafter accepts nearly all of it, and a 128-token run mostly measures the width
  schedule's opening regime. On real mid-session contexts the same engine measures 31.7
  tok/s median / 21.4 floor greedy and 27.6 / 17.7 at the thinking preset (the
  [speculative-decoding tables](#two-throughput-levers) below), which is ~2× serial and
  the number a session actually lives at.
- The per-step cost in an agent loop is what no single-shot benchmark shows. Both
  engines can reuse a prompt prefix; the difference is that chad keeps the transcript a
  strict token-prefix of the live cache *by construction*, across compaction and across
  sessions. That is the [next section](#the-agentic-loop-win-055-s-per-step-not-48-s).

## The agentic-loop win: ~0.55 s per step, not ~48 s

The headline is what a *follow-up* turn costs, not the cold-prefill rate. On the shipped
model a 5,000-token transcript prefills cold in **~48 s**. The next agentic step
only appends the model's reply, a tool call, and the tool's output, so with the persistent
prefix cache it re-reads **nothing**: the follow-up turn prefills just the ~16 appended
tokens in **~0.55 s**.

```
cache-less backend:  re-prefill all 5,143 tokens  ->  ~48 s of dead air, every step
chad (prefix cache): prefill the 16 new tokens     ->  ~0.55 s, every step
```

That ~85× gap is the entire reason a local model can feel like an agent instead of a batch
job, and it widens with the transcript, since the cache-less side grows while the warm step
stays flat. Note which side of the trade the slow cold prefill lands on: it is paid once per
divergence, and the cache is what makes it once. Why that cache is *append-only* (and why
that's the right trade for a hybrid SSM/attention model) is in
[the cache trade](design.md#trimmable-vs-append-only-the-cache-trade-chad-lives-with).

## The second session in a project starts warm

The ~0.55 s figure above is the *within*-session win. Across sessions there is a second one,
and it is larger: chad checkpoints the stable system+tools KV prefix to disk
(`engine.warm_prefix`) and reloads it when you next start in the same project, so the
~2.8k-token system+tools prefix is prefilled **once, ever** rather than once per session.
(That prefix is the byte-identical head of every turn: the behavioral prompt, the workspace
snapshot, and the five tool schemas. It carries no skills catalog, which is most of why it is
~2.8k and not the ~8k a tier-1 disclosure would cost; see
[Agent Skills](configuration.md#agent-skills-agentskillsio).)

Measured on the shipped 27B (M4 Pro, 24 GB), one fixture and one ask, cold vs. warm:

| | cold (first session in a project) | warm (checkpoint hit) |
|---|---|---|
| ask → first tool call | 75.6 s | **5.5 s** |
| whole turn | ~103 s | **31.4 s** |

The cold column is a real cost and worth stating plainly: the first turn in a *new* project
spends over a minute reading a system prompt before it does anything you asked for. Until
2.0.3 every new directory paid it, because the checkpoint was keyed on the whole system
prompt, cwd and workspace listing included, and a fresh directory could never hit it
(`benchmarks/matrix` measured 32 of 32 fresh-directory cells missing, 24 s each; on 2.0.3
three nights of the same grid have 47 of 48 restoring the head and prefilling a
~320-token tail in 3.2-3.6 s). Now there are two checkpoints: the full prefix, which a restart in the same
project restores outright, and its project-independent head (the tool schemas and
behavioral prompt, most of the prefix), which any directory restores before prefilling
only its own cwd/listing/docs tail.
Both survive restarts and are invalidated exactly when the text they cache changes. The
session banner's `[warm start: N prefix tokens from disk cache]` line (or
`[…; M project tokens prefilled in S s]` for the head-only case) is chad telling you which
of these turns you are about to have.

## Why decode sits where it does

The first-order answer is memory bandwidth: each token streams the resident weights through
the chip once, so `tok/s ≈ bandwidth / resident-bytes-per-token`. On a **dense** model that
story is largely right, and it is why the quant is as aggressive as it is: every parameter
is read for every token, so shrinking the weights is the only decode lever there is. ~12 GB
(the 3-bit) against this M4 Pro's ~273 GB/s is the envelope the engine works inside, and no
amount of kernel work moves that wall. The ternary build's 7.2 GB should sit further inside
it than it does: mlx's 2-bit GEMV streams at ~140 GB/s, about half the roofline, which is
why its serial step (21 tok/s) is not the 30+ the byte count implies, and why the drafter
carries the speed story there.

Inside the envelope, two things decide how close you get. Both were first measured on the
retired 35B, whose sparse MoE made them unmissable; the *lessons* are what carried into
2.0.0, and the code that serves them was rebuilt for the dense checkpoint.

- Dispatch cost is real, and work that removes kernels pays. In-situ ablation of the
  35B's decode step put bandwidth-minimal cost around 9 ms against ~14 ms real: the step
  issued roughly **400 Metal kernels per token**, each carrying ~9 µs of launch and gap
  latency. A step that is waiting on the command queue does not care how few bytes you
  read. chad's decode fast-path
  ([`mlx_fastpath.py`](../src/chad/mlx_fastpath.py)) attacks the kernel count directly: on
  the shipped model it concatenates each layer's MLP `gate|up` pair and the GatedDeltaNet's
  four same-input `in_proj` tensors into single `quantized_matmul`s, then compiles the whole
  S=1 layer step, so the MLP block and GDN forward are each one call with layernorms and
  residuals folded in. Row-wise math is unchanged and greedy token choices were verified identical to stock.
  Prefill deliberately keeps the stock op graph: the compiled kernels change bf16 rounding,
  and on a recurrent hybrid a prefill-side rounding change compounds across the whole
  transcript. `CHAD_NO_FASTPATH=1` is the A/B arm.
- Attention is a reuse problem. Ablating the math out of the fused
  quantized-KV kernel leaves it streaming at ~331 GB/s, already at this machine's measured
  roofline, with ~68% of its runtime spent on the GQA q-heads re-reading staged K/V out of
  threadgroup memory. That is why [`mlx_qsdpa.py`](../src/chad/mlx_qsdpa.py) has a
  `simdgroup_matrix` schedule giving each simdgroup its own position-stream instead of its
  own head. This is the part of the kernel work that survived the model change intact: the
  same kernel serves the shipped model, and it now carries a multi-token (S>1) tier so
  speculative verification and prefill get the fused path too, worth 8% at 8k context and
  30% at 38k over the dequantize-the-whole-cache fallback at the widths the draft schedule
  actually jumps to.

The honest caveat, and the reason every number on this page is end-to-end rather than
per-kernel: **isolated kernel speedups oversell badly.** The attention retile measured
1.26-1.40× *on the kernel* and moved the whole decode step ~2%, because attention is only a
fifth of it. Trust steady-state `chad-bench` tok/s; don't extrapolate from a microbenchmark.

You don't have to tune any of this. chad picks the fast configuration at startup. The
KV-cache bit width, the fused-attention schedule (which of its kernels runs, up to what
verify width, and the split factor, which widens past 16k context except for the one kernel
whose partials slab makes that a loss), and the speculative draft depth are all chosen from
measurements and applied for you. MLX's own runtime knobs are deliberately left alone:
`MLX_METAL_FAST_SYNCH` and the command-buffer size limits were swept and every setting was
*slower* than mlx's defaults, so chad overrides none of them. `chad-bench` reports what
you're getting.

## The model: Qwen3.8-27B

chad runs **one** model, [`Qwen3.8-27B Ternary-Bonsai-2-DFlash2`](https://huggingface.co/nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX):
a dense `qwen3_5` hybrid (64 layers: 48 GatedDeltaNet + 16 full attention) in Prism ML's
ternary build, every projection Hadamard-rotated and stored as 2-bit group-128 with levels
{−s, 0, +s}, ~7.2 GB resident, 262k native context. One model, every machine, with no
RAM-aware pick, no size tier and no flag. chad targets 24 GB Macs and nothing smaller. The
vision tower is not present at all (dropped in the repack; the text-only loader would have
skipped it anyway). The 3-bit
[`UD-Q3_K_XL-DFlash2`](https://huggingface.co/nathansutton/Qwen3.8-27B-UD-Q3_K_XL-DFlash2-MLX)
quant of the same model is one `--model` away and is the quality reference below.

It's a *thinking* model that emits tool calls in the XML `<function=…>` dialect; the harness
parses both that and JSON, and strips `<think>` blocks. The repo also carries the
**DFlash2 block drafter** it decodes with, pre-quantized in `dflash/`; see
[speculative decoding](configuration.md#speculative-decoding--kernel-knobs). There are no
model flags to pick from; you just run `chad`.

### Ternary against the 3-bit

What the ternary build costs and buys against the 3-bit quant of the same model. See
[configuration](configuration.md#what-the-engine-does-with-the-ternary-weights) for what
chad does to run it. Measured on the M4 Pro, one load per process, greedy, 512-token prompt,
128-token decodes; "as loaded" is the upstream pack's own forward with nothing of chad's
attached (fp32 activations, stock 2-bit matmul, no drafter):

| | 3-bit (`--model`) | ternary, as loaded | ternary, shipped |
|---|---|---|---|
| weights resident | 12.33 GB | 7.15 GB | 7.15 GB |
| serial decode | 17.9 tok/s | 17.8 | **21.3** |
| drafted decode (DFlash2) | ~60 tok/s | 12.1 (net loss) | **63.7** (94% accepted) |
| width-8 verify forward | 2.2 × a step | 9.2 × | **2.2 ×** |
| prefill, 5k prompt | ~104 tok/s | 89 | 99 |
| peak at 5k / 32k | — / 16.8 GB | — / 13.3 GB | 11.8 GB / 15.5 GB³ |
| governor window (24 GB) | ~56k tokens | — | **~150k** (114k with the bench's residue resident) |
| code NLL / ppl (1,535 tok, teacher-forced) | **1.383 / 3.99** | 1.500 / 4.48 | 1.502 / 4.49 |
| private eval tiers (core, languages, hard, brutal, realworld, recovery) | 56/56 | — | 56/56 |

The perplexity row is the honest cost: +12% on code against the 3-bit, which the task-level
tiers do not resolve (56 tasks, no flip in either direction, at the shipped sampling preset). The window row is what it buys, and on this agent the window is what
runs out first.

³ With the drafter resident and after the bench's earlier sections (active 9.6 GB going
in, against 8.25 GB at a clean load); the 32k prefill ran at 90 tok/s and decode at 32k
drafted 62 tok/s, no depth falloff. Running the compiled bodies' rotation in fp32
(`CHAD_PRISM_ROT_FP32=1`) measured 63.6 drafted / 21.0 serial against 63.7 / 21.3: inside
the noise, so the one-kernel native-dtype rotation stays the default. That
rotation is what every path runs, prefill included (the sign vectors fold into the weights
at install rather than inside the compiled bodies); teacher-forced code NLL over 7k tokens
of three files reads 1.3830 / 1.6812 / 1.3576 native against 1.3847 / 1.6827 / 1.3573 with
the fp32 transform, and prefill is 1-3% faster.

The "as loaded" drafted number is the S=2..8 wall: the stock 2-bit matmul re-pays the
weight read per verify row, so an 8-wide verify cost 9× a decode step and never amortized.
The small-M MMA kernel at 2-bit g128 is flat in width (0.35 ms at M=8 on the 5120→17408
projection against 0.71 ms stock), which puts the block on the right side of the cliff.
The serial step is bounded by mlx's 2-bit GEMV rate, not by dispatch, so the fast-path's
kernel-count win is smaller here than on the 3-bit; the weights, the verify and the
context are where the pack pays.

The same cliff sat one row past the tile. What one main-model forward of width M costs
(`benchmarks/verify_ladder.py`: one load, 8-bit KV, 2k tokens in front, median of 8):

| width M | 1 | 2 | 4 | 8 | 9 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| one tile only, ms | 49.3 | 68.3 | 109.1 | 108.2 | 266.3 | 317.1 | 342.9 | 354.7 | 357.3 |
| tiled (shipped), ms | 49.2 | 67.2 | 107.7 | 106.8 | **147.2** | **203.6** | **203.4** | **297.2** | 353.1 |
| shipped, serial steps | 1.00 | 1.37 | 2.19 | 2.17 | 2.99 | 4.14 | 4.14 | 6.04 | 7.18 |

Widths 9-24 are several 8-row kernel calls now (1.2-1.8×), which is where an agent step's
warm tail and a short tool-result suffix land; past 24 mlx's own tiling wins and keeps the
forward. Widths 1-8 moved ~1%: folding the sign vectors took ~500 kernel launches out of a
verify forward and it barely shows, because a verify forward is matmul-bound, not
dispatch-bound — the same thing a compiled verify step measured on the 3-bit.

A drafted **round** costs more than its verify forward: the drafter's own forward, the
rollback of rejected positions and the host sync ride on top, and on a 46-51 ms serial
step they are a larger share than on the 3-bit's 57. Round wall over the serial step, a
fixed width per arm (`benchmarks/spec_decode.py`, six prompt × preset cells agreeing to
±0.05 within a run; the ratio moves with the machine's serial step, so two runs):

| drafts verified | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| the 3-bit's ladder | 1.0 | 1.76 | 1.93 | 2.30 | 2.18 | 2.20 | 2.19 | 2.20 |
| ternary, cool (21.5 tok/s serial) | 1.0 | 1.68 | 2.21 | 2.60 | 2.56 | 2.57 | 2.57 | 2.60 |
| ternary, warm (19.7 tok/s serial) | 1.0 | 1.58 | 2.09 | 2.44 | 2.39 | 2.44 | 2.44 | 2.46 |

The schedule picks its width from this ladder, and the 3-bit one prices every width past
one 9-18% too cheap here, so the shipped schedule is seeded with the midpoint of the two
measured rows and carries what each turn measures into the next. What that is worth in
tok/s is not resolved: on the low-acceptance prose these rows come from (20-30% of drafted
positions accepted) the schedule sits at break-even with serial on either seed (19.1-20.9
tok/s against 19.0-19.9 serial, both trees, back to back on an idle machine), inside the
±1 tok/s run-to-run noise of three prompts.

## Two throughput levers

**Thinking budget.** This is a reasoning model, and its `<think>` blocks dominate what it
generates: measured on real agentic traces, **~62-66% of all generated tokens**. Because
decode runs at a roughly fixed tok/s whatever it is writing, those tokens are pure
wall-clock. `--no-think` (a per-turn toggle, or the flag) injects an empty think block and
skips them, which is the most effective time-to-done lever on well-scoped agentic work.
Thinking stays **on** by default, since it helps on harder reasoning; flip it off when the
task is well-specified and you'd rather not wait on the reasoning tokens.

**Speculative decoding.** chad drafts with a DFlash2 block drafter, a 1.9B model that reads
the main model's residual stream at five layers and proposes a block of tokens in one
forward, then verifies in one batched main-model forward, accepting by exact rejection
sampling: every emitted token is the model's own choice, sampled output keeps its true
distribution, and greedy output matches plain decoding to kernel rounding (it follows serial
until the first near-tie, since the batched verify and the serial step run different
kernels). Measured on
an M4 Pro with the shipped quant (one load, same prompts, 384-token decodes):

| sampling | serial | **DFlash2** |
|---|---|---|
| greedy | 17.5 tok/s | **60.1 (3.4×)** |
| thinking preset (temp 1.0, top_p 0.95, top_k 20, the default) | 22.4 | **49-51** |
| non-thinking preset (temp 0.7, top_p 0.80, top_k 20) | 23.7 | **52-54** |

Ten prompts (eight ~512-token prose seeds, two code continuations), 384-token decodes, medians,
one load per run, on a 3-bit g64 quant of the same model. On prose the drafter lands ~8 tokens
per round; code runs ~44 tok/s greedy. Those seeds are public-domain text the model has
memorized, so they show the ceiling. **On real traffic** (`benchmarks/spec_decode.py`, ten
mid-session contexts of 12-19k tokens replayed out of `~/.chad/sessions` with their tool
results and schemas, 384-token decodes) acceptance inside `<think>` is 35-55% and the numbers
are:

| sampling (real contexts) | serial | schedule (default) | fixed full block |
|---|---|---|---|
| greedy (median / floor) | 14.8 / 13.2 | **31.7** / **21.4** | 36.0 / 17.5 |
| thinking preset (median / floor) | 13.9 / 13.1 | **27.6** / **17.7** | 27.1 / 15.7 |

The per-round width schedule is the default because of the floor column: a full-block round
costs ~2.2 serial steps whatever it commits, so on the worst prompt a fixed block drops to
17.5 tok/s (and to 11.7, below serial, on a repo-text seed at the thinking preset) while the
schedule narrows and never measured under serial. `CHAD_DFLASH_ADAPTIVE=0` is the fixed arm. Two pieces make that number: the block drafter's
acceptance, and the small-M matmul kernel (`mlx_qmm_mma.py`) that makes an 8-row verify cost
what a 5-row one does: a full-block round is 2.2 serial steps, flat from width 4 up. Without
the kernel each extra verify row costs ~33 ms and the drafter's optimum is width 4 at 28 tok/s;
with its first, threadgroup-staged version the round cost 3.2 steps (47.7 / 41.4). The wall
now is the chip's MMA issue rate, not memory. `CHAD_NO_DFLASH=1` turns speculation off
entirely and decodes one token per forward, which is the `serial` column.

Prompt-lookup decoding (PLD), the draft-model-free variant that proposes continuations from
n-grams already in context, is implemented and greedy-exact but **opt-in**
(`CHAD_USE_PLD=1`). It drafts from context *recurrence*, and on real agentic traces that is a
minority of what this agent writes: whole-session contribution measured at **+2.2%** of
generated tokens. It also doesn't compose with block speculation (one generate loop each), so
where the drafter is available it is strictly the better of the two.

## Nine harnesses, one laptop: what the agent loop costs

The rows above are engine numbers: how fast one process reads and writes tokens. They say
nothing about what a person feels once an agent loop sits on top, and that turned out to
be decided by the harness, not the engine. `benchmarks/matrix/` runs nine coding agents
(pi, opencode, chad, deepseek-harness, goose, mini-swe-agent, crush, cline, codex) against
**one** `llama-server` on the same GGUF, same eight Exercism tasks, same sampler forced on
every request by a proxy, and records from the server's side what each harness made it
read: the first-turn prompt, the uncached tokens per later turn, the wait each of those
cost, the cache-reuse rate, and the side requests fired beside the agent loop. Same
harness on chad's MLX engine is the engine cell.

The committed run (`benchmarks/matrix/_runs/`), its method, its versions and its caveats
are in [`benchmarks/matrix/README.md`](../benchmarks/matrix/README.md). The short form:
on this laptop the first token arrives 13 s or 238 s after you press enter depending on
the harness, and a later turn waits 1 s or 39 s, on the same weights.

---

*Day-to-day correctness is tracked in a private eval suite (it seeds repos, runs the agent,
and verifies the actual edit) and is not quoted here. This page stays focused on the numbers
you can reproduce yourself: `chad-bench` and `benchmarks/stock/`.*
