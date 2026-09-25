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
| **Qwen3.8-27B `UD-Q3_K_XL` GGUF** (shipped, default) | 104 tok/s | 46.1 tok/s | 0.51 s (16 tok) |
| same, serial (`CHAD_NO_DFLASH=1`) | 105 tok/s | 11.6 tok/s | 0.51 s (16 tok) |

> Measured on this machine with the command above, chad 2.2.0 at `ca0598a`. Run it on
> yours; these are hardware numbers, not scores.

Prefill is the honest cost of a dense checkpoint: every one of the 27B parameters is read
for every token of the prompt, so ~105 tok/s is close to this chip's compute roofline rather
than a tuning failure. It is also why the [warm prefix
cache](#the-second-session-in-a-project-starts-warm) matters more than any decode work: the
third column, not the first, is what a session actually pays after its first turn.

The default-row decode figure flatters the drafter: `chad-bench` tiles a block of code, the
drafter accepts nearly all of it, and a 128-token run is short. Across the 36 agent tasks
of a published polyglot run ([`q3kxl-chad-h36`](../benchmarks/polyglot/RUNS.md)) the median
trial decoded at 21.2 tok/s, and that is the number a session lives at.
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
file, Unsloth's GGUF, which both engines now read as it is, on the same M4 Pro (24 GB), one
engine resident at a time, each measured with its own native benchmark on a 512-token
prompt and a 128-token generation. Ollama was measured once on the same 
GGUF (0.32.15, a `FROM`-only Modelfile, `num_ctx` 2048, temperature 0, timed from 
`/api/generate`'s own counters) at 96 tok/s prefill and 10.9 tok/s decode, the llama.cpp 
number, as expected. The row is kept under `_runs/ollama.json`. There is no script arm 
because importing a GGUF into Ollama needs ~45 GB of scratch disk for nothing new.

| Engine | Prefill (512-tok prompt) | Decode (128 tok) | Speculative decoding |
|---|---|---|---|
| llama.cpp `llama-bench` (stock, build 10470) | 102 tok/s | 10.9 tok/s | off in this benchmark |
| llama.cpp `llama-server` (build 10917), serial | 97 tok/s | 11.3 tok/s | off |
| llama.cpp `llama-server` (build 10917) | 95 tok/s | 11.1 tok/s | DFlash2 drafter (Q4_K_M GGUF) |
| **chad**, serial (`CHAD_NO_DFLASH=1`) | 99 tok/s | 11.8 tok/s | off |
| **chad**, default | 102 tok/s | **48.3 tok/s** | DFlash2 block drafter |

Reproduce it with `uv run python benchmarks/stock/stock.py llama`, `… llama-dflash` and
`… chad`, one arm at a time since each loads ~13 GB, then `… table` to render the rows.
The two `llama-server` rows are one build and one instrument, serial and drafted, on the
same 512-token prompt `chad-bench` tiles; `llama-dflash` needs build 10658 or later
(`STOCK_LLAMA_BIN` points at an unpacked release without touching a brew install). The measured rows are
committed under `benchmarks/stock/_runs/`. How to read it:

- Prefill is a wash. Both engines read a 512-token prompt at ~100 tok/s: a dense 27B
  reads every parameter for every prompt token, and that is the chip's compute roofline,
  not anyone's tuning.
- Serial decode is a wash too. Same bytes, same bandwidth wall: 11.8 against 11.3 tok/s.
  chad's fused single-token step (below) buys little here, because decoding the GGUF's
  codebook formats, not kernel dispatch, is what bounds it. An MLX affine repack of the
  same model decoded 17.9 tok/s serially and passed 6 of 9 agent trials where this file passes 9
  ([design](design.md#the-weights)).
- The drafter is the gap that matters, and what decides it is the verify pass.
  llama.cpp has run DFlash2 drafters since build 10658 (`--spec-type draft-dflash`), and
  the drafter is published in GGUF form (`incoai/Qwen3.8-27B-DFlash2-GGUF`, BF16 / Q8_0 /
  Q4_K_M). On this checkpoint it works as a drafter, 96.5% of drafted tokens accepted on
  the tiled prompt, and decodes at 11.1 tok/s against 11.3 serial anyway. A round verifies
  8 tokens in one batch, and llama.cpp's Metal path runs a batch of 1 / 2 / 4 / 8 / 16 at
  11.2 / 12.8 / 13.5 / 14.0 / 45.5 tok/s (`llama-bench -p 1,2,4,8,16 -n 0`, recorded in
  `_runs/llama-dflash.json`): the 8-token verify alone costs ~6.4 serial steps, so no
  acceptance rate can pay for the round, and the jump at 16 is a kernel switch the round
  sits just below. chad's whole round at the same width, drafter included, costs ~2.0
  serial steps (`mlx_dflash.BLOCK_ROUND_COSTS_GGUF`), which is the difference between the
  two drafted rows.
  llama.cpp's DFlash2 PR reports ~1.8× on an M5 Pro with a Q4_K_M target, so this measures
  this GGUF on this Mac, not llama.cpp's DFlash2 in general.
- 48 is a ceiling, and a session runs slower. `chad-bench` tiles a block of code and the
  drafter accepts nearly all of it. Across 36 real agent tasks the median trial decoded at
  21.2 tok/s, ~2× serial, and that is what a session lives at.
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
[the cache trade](design.md#the-cache-only-appends).

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
(the [nine-harness grid](#nine-harnesses-one-laptop-what-the-agent-loop-costs) measured 32 of 32 fresh-directory cells missing, 24 s each; on 2.0.3
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
is read for every token, so shrinking the weights is the only decode lever there is. The
shipped 13.2 GB against this M4 Pro's ~273 GB/s is the envelope the engine works inside,
and no amount of kernel work moves that wall. The GGUF's codebook formats cost more to
decode than an affine group does, so its serial step sits at ~13 tok/s, under the
envelope, and the drafter carries the speed story.

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
  four same-input `in_proj` tensors into single matmuls (on a GGUF, wherever the partners
  share a block format), then compiles the whole
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

chad runs **one** model: Qwen3.8-27B, a dense `qwen3_5` hybrid (64 layers: 48 GatedDeltaNet
+ 16 full attention, 262k native context), as Unsloth's
[`UD-Q3_K_XL`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) GGUF, read natively,
~13.2 GB resident. It's a *thinking* model that emits tool calls in the XML `<function=…>`
dialect; the harness parses both that and JSON, and strips `<think>` blocks. The
**DFlash2 block drafter** comes from a small sidecar repo; see
[speculative decoding](configuration.md#speculative-decoding--kernel-knobs). There are no
model flags to pick from; you just run `chad`.

### The ternary pack, and why it is no longer the default

2.x shipped Prism ML's ternary build: every projection Hadamard-rotated and stored as 2-bit
with levels {−s, 0, +s}, 7.2 GB. Against the affine 3-bit repack of the same model, one
load per process, greedy, 512-token prompt:

| | 3-bit repack | ternary |
|---|---|---|
| weights resident | 12.33 GB | 7.15 GB |
| serial decode | 17.9 tok/s | 21.3 |
| drafted decode (DFlash2) | ~60 tok/s | 63.7 |
| prefill, 5k prompt | ~104 tok/s | 99 |

We read that table as the ternary winning: as fast, 5 GB lighter, and a much bigger
window, with a quality cost our task set at the time was too easy to show. On 36 polyglot
tasks it passed 24 where Unsloth's GGUF passes 34, and all twelve failures ran out the wall
clock ([`tern-chad-h36`, `q3kxl-chad-h36`](../benchmarks/polyglot/RUNS.md)). [Design](design.md#the-weights) has the account;
[configuration](configuration.md#running-the-ternary-pack) has what chad does to run the
pack, which still loads with `--model`.

## Two throughput levers

**Thinking budget.** This is a reasoning model, and its `<think>` blocks dominate what it
generates: **65% of all generated tokens** across the 36 tasks of
[`q3kxl-chad-h36`](../benchmarks/polyglot/RUNS.md). Because
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
kernels).

On the shipped GGUF a drafted round costs ~2.0 serial steps from depth 2 to a full block
of 8 (`mlx_dflash.BLOCK_ROUND_COSTS_GGUF`): the verify decodes each weight block once into
register-operand MMAs for every row, where a stock matmul would re-read it per row. That
flat round is what the drafter's 46-48 tok/s on tiled code and 21.2 tok/s median on real
agent tasks (against 11.6 serial) stand on. A per-round width schedule picks how far to
draft from the acceptance it has been seeing, so a low-acceptance stretch narrows instead
of paying for a full block it will mostly reject; `CHAD_DFLASH_ADAPTIVE=0` is the
fixed-width arm, and `CHAD_NO_DFLASH=1` turns speculation off.

Prompt-lookup decoding (PLD), the draft-model-free variant that proposes continuations from
n-grams already in context, is implemented and greedy-exact but **opt-in**
(`CHAD_USE_PLD=1`). It drafts from context *recurrence*, and on real agentic traces that is a
minority of what this agent writes: whole-session contribution measured at **+2.2%** of
generated tokens. It also doesn't compose with block speculation (one generate loop each), so
where the drafter is available it is strictly the better of the two.

## Nine harnesses, one laptop: what the agent loop costs

The rows above are engine numbers: how fast one process reads and writes tokens. They say
nothing about what a person feels once an agent loop sits on top, and that turned out to
be decided by the harness, not the engine. A grid ran nine coding agents
(pi, opencode, chad, deepseek-harness, goose, mini-swe-agent, crush, cline, codex) against
**one** `llama-server` on the same GGUF, same eight Exercism tasks, same sampler forced on
every request by a proxy, and records from the server's side what each harness made it
read: the first-turn prompt, the uncached tokens per later turn, the wait each of those
cost, the cache-reuse rate, and the side requests fired beside the agent loop. Same
harness on chad's MLX engine is the engine cell.

The grid is archived: its runner, method, versions, caveats and every row it recorded are at
the tag [`archive/matrix-nine-harnesses`](https://github.com/nathansutton/chad/tree/archive/matrix-nine-harnesses/benchmarks/matrix). The short form:
on this laptop the first token arrives 13 s or 238 s after you press enter depending on
the harness, and a later turn waits 1 s or 39 s, on the same weights.

---

*Correctness is measured by [`benchmarks/polyglot`](../benchmarks/polyglot/README.md): 215
exercises in six languages, run by the shipped agent on the laptop. Its pass rates are not
quoted here, because its job is telling two builds apart with a paired test, not ranking
chad. This page stays focused on the numbers you can reproduce yourself: `chad-bench` and
`benchmarks/stock/`.*
