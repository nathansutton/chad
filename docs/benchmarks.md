# Throughput & performance

Three numbers decide whether a local agent feels responsive: prefill speed, decode speed,
and the per-step cost once the cache is warm. Every number here is a hardware measurement
you can reproduce on your own Mac. Nothing here is a score against anyone else: chad's
premise is a 24 GB laptop, and the only comparisons that honour it are between engines on
that laptop. The engineering behind the numbers is in [design](design.md).

## Reproduce it yourself

```bash
uv run chad-bench                       # default: 5000-token prefill, 128-token decode
uv run chad-bench --prefill-tokens 8000 --gen-tokens 256
uv run chad-bench --chunk 1024          # force a prefill chunk size
uv run chad-bench --agentic             # the truncated-turn cache miss, below
```

It drives the real engine on the real model and reports three things: cold prefill (how
fast the model reads a fresh prompt, the bill a naive loop pays every step), decode (how
fast it writes, bandwidth-bound and roughly constant), and the warm step (how few tokens
a follow-up turn has to prefill once the prefix cache is warm).

`--agentic` seeds a large context (`--context-tokens`, default 24,000) and reproduces the
case where a step ends mid-`<think>` and the next prompt is no longer an extension of the
cache, reporting the re-prefill that step pays with and without the fix.

## Measured throughput (M4 Pro, 24 GB)

5,000-token cold prompt, 128-token decode, a follow-up turn that appends ~16 tokens:

| Model | Prefill (cold) | Decode | Warm-step prefill |
|---|---|---|---|
| **Qwen3.8-27B `UD-Q3_K_XL` GGUF** (shipped) | 104 tok/s | 46.1 tok/s | 0.51 s (16 tok) |
| same, serial (`CHAD_NO_DFLASH=1`) | 105 tok/s | 11.6 tok/s | 0.51 s (16 tok) |

> Measured with the command above on chad 2.3.0 (`ca0598a`). Run it on yours; these are
> hardware numbers, not scores.

Prefill is the honest cost of a dense checkpoint: every one of the 27B parameters is read
for every prompt token, so ~105 tok/s is this chip's compute roofline, not a tuning
failure. It falls as context grows, since attention is quadratic. Decode is nearly flat.
The warm step is a property of the prefix cache, not the weights, and it is the number
that decides how the agent feels.

The drafted decode figure is a ceiling. `chad-bench` tiles a block of code and the drafter
accepts nearly all of it. Across the 36 agent tasks of a published polyglot run
([`q3kxl-chad-h36`](../benchmarks/polyglot/RUNS.md)) the median trial decoded at
21.2 tok/s, and that is the number a session lives at.

## Same model, same Mac, stock engine

What chad's engine buys over a generic tool pointed at the same weights on the same
laptop. Both read Unsloth's `UD-Q3_K_XL` GGUF, and both can speculate with the same
DFlash2 drafter. One engine resident at a time, each measured with its own native
benchmark on a 512-token prompt and a 128-token generation:

| Engine | Prefill | Decode | Speculative decoding |
|---|---|---|---|
| llama.cpp `llama-bench` (build 10470) | 102 tok/s | 10.9 tok/s | off |
| llama.cpp `llama-server` (build 11184), serial | 96 tok/s | 11.4 tok/s | off |
| llama.cpp `llama-server` (build 11184) | 95 tok/s | 11.1 tok/s | DFlash2 drafter (Q4_K_M GGUF), 96.5% accepted |
| **chad**, serial (`CHAD_NO_DFLASH=1`) | 102 tok/s | 11.7 tok/s | off |
| **chad**, default | 101 tok/s | **48.2 tok/s** | DFlash2 block drafter |

Ollama was measured once on the same GGUF (0.32.15, `FROM`-only Modelfile, temperature 0)
at 96 / 10.9 tok/s, the llama.cpp number as expected; it runs llama.cpp underneath.
Reproduce the rows with `benchmarks/stock/stock.py` ([README](../benchmarks/stock/README.md));
the measured rows are committed under `benchmarks/stock/_runs/`.

- **Prefill is a wash.** A dense 27B reads every parameter for every prompt token on
  either engine.
- **Serial decode is a wash.** Same bytes, same bandwidth wall. Decoding the GGUF's
  codebook formats, not kernel dispatch, is what bounds it.
- **The drafter is the gap, and the verify pass decides it.** llama.cpp runs the same
  drafter and accepts 96.5% of what it proposes, yet decodes no faster, because its Metal
  path runs an 8-token verify batch at ~6.4× the cost of a serial step (`llama-bench -p
  1,2,4,8,16 -n 0`: 11.3 / 12.8 / 13.6 / 14.0 / 45.5 tok/s, recorded in
  `_runs/llama-dflash.json`), so no acceptance rate can pay for the round. chad's whole
  round at the same width costs ~2.0 serial steps. llama.cpp's DFlash2 PR reports ~1.8× on
  an M5 Pro with a Q4_K_M target, so this measures this GGUF on this Mac, not llama.cpp's
  DFlash2 in general.
- **The per-step cost in an agent loop is what no single-shot benchmark shows.** Both
  engines can reuse a prompt prefix; chad keeps the transcript a strict token-prefix of
  the live cache by construction, across compaction and across sessions.

## The agentic-loop win: ~0.55 s per step, not ~48 s

On the shipped model a 5,000-token transcript prefills cold in ~48 s. The next step only
appends the model's reply, a tool call and its output, so with the prefix cache the
follow-up turn prefills just the ~16 appended tokens in ~0.55 s.

```
cache-less backend:  re-prefill all 5,143 tokens  ->  ~48 s of dead air, every step
chad (prefix cache): prefill the 16 new tokens     ->  ~0.55 s, every step
```

That ~85× gap is the reason a local model can feel like an agent instead of a batch job,
and it widens with the transcript. Why the cache is append-only is in
[design](design.md#the-cache-only-appends).

## The second session in a project starts warm

Across sessions there is a second win, and it is larger: chad checkpoints the system+tools
KV prefix to disk and reloads it when you next start, so the ~2.8k-token prefix is
prefilled once, ever. (It carries no skills catalog, which is most of why it is ~2.8k and
not ~8k; see [Agent Skills](configuration.md#agent-skills-agentskillsio).) Measured on the
shipped model, one fixture and one ask:

| | cold (first session in a project) | warm (checkpoint hit) |
|---|---|---|
| ask → first tool call | 75.6 s | **5.5 s** |
| whole turn | ~103 s | **31.4 s** |

There are two checkpoints: the full prefix, which a restart in the same project restores
outright, and its project-independent head (the tool schemas and behavioural prompt), which
any directory restores before prefilling only its own cwd, listing and docs tail, ~320
tokens in 3.2–3.6 s. Both survive restarts and are invalidated exactly when the text they
cache changes. The banner's `[warm start: N prefix tokens from disk cache]` line (or
`[…; M project tokens prefilled in S s]` for the head-only case) says which turn you are
about to have.

## Why decode sits where it does

Each token streams the resident weights through the chip once, so
`tok/s ≈ bandwidth / resident bytes`. On a dense model that story is right, which is why
the quant is aggressive: shrinking the weights is the only decode lever there is. 13.2 GB
against this M4 Pro's ~273 GB/s is the envelope, and no kernel work moves it. The GGUF's
codebook formats cost more to decode than an affine group, so the serial step sits under
the envelope at ~12 tok/s, and the drafter carries the speed story.

Inside the envelope, two things decide how close you get:

- **Dispatch cost is real.** A decode step issues hundreds of Metal kernels per token,
  each with ~9 µs of launch and gap latency, and a step waiting on the command queue does
  not care how few bytes you read. The fast path (`mlx_fastpath.py`) concatenates each
  layer's MLP `gate|up` pair and the GatedDeltaNet's four same-input projections into
  single matmuls and compiles the S=1 layer step. Row-wise math is unchanged and greedy
  token choices were verified identical to stock. Prefill keeps the stock op graph, since
  compiled kernels change bf16 rounding and on a recurrent hybrid that compounds across
  the transcript. `CHAD_NO_FASTPATH=1` is the A/B arm.
- **Attention is a reuse problem.** With the math ablated out, the fused quantized-KV
  kernel streams at ~331 GB/s, this machine's roofline, and ~68% of its runtime is the GQA
  q-heads re-reading staged K/V. So `mlx_qsdpa.py` gives each simdgroup its own position
  stream instead of its own head, and carries a multi-token tier so speculative
  verification and prefill get the fused path too: 8% at 8k context and 30% at 38k over
  the dequantize-the-cache fallback.

Isolated kernel speedups oversell badly: the attention retile measured 1.26–1.40× on the
kernel and moved the decode step ~2%, because attention is a fifth of it. Trust
steady-state `chad-bench` tok/s, not microbenchmarks. You do not tune any of this: chad
picks the KV bit width, attention schedule and draft depth at startup from measurements.
MLX's own runtime knobs are left alone, since every setting of `MLX_METAL_FAST_SYNCH` and
the command-buffer limits measured slower than the defaults.

## Two throughput levers

**Thinking.** This is a reasoning model, and `<think>` is 65% of what it generates across
the 36 tasks of `q3kxl-chad-h36`. Decode runs at a fixed tok/s whatever it is writing, so
those tokens are pure wall clock. `--no-think` skips them and is the most effective
time-to-done lever on well-scoped work; thinking stays on by default because it helps on
harder tasks.

**Speculative decoding.** A DFlash2 block drafter, a 1.9B model that reads the main
model's residual stream at five layers, proposes a block of tokens in one forward. The
block is verified in one batched main-model forward and accepted by exact rejection
sampling, so every emitted token is the model's own choice. On the shipped GGUF a round
costs ~2.0 serial steps from depth 2 to a full block of 8, because the verify decodes
each weight block once for every row where a stock matmul re-reads it per row. A
per-round schedule narrows the draft when acceptance drops; `CHAD_DFLASH_ADAPTIVE=0` is
the fixed-width arm and `CHAD_NO_DFLASH=1` turns speculation off. Prompt-lookup decoding
is implemented but opt-in (`CHAD_USE_PLD=1`): it drafts from context recurrence, measured
at +2.2% of generated tokens, and does not compose with the block drafter.

## Nine harnesses, one laptop

The rows above are engine numbers. What a person feels once an agent loop sits on top
turned out to be decided by the harness: a grid ran nine coding agents (pi, opencode,
chad, deepseek-harness, goose, mini-swe-agent, crush, cline, codex) against one
`llama-server` on the same GGUF and the same eight Exercism tasks, recording from the
server's side what each harness made it read. On this laptop the first token arrives 13 s
or 238 s after you press enter depending on the harness, and a later turn waits 1 s or
39 s, on the same weights. The grid, its method and every row are archived at the tag
[`archive/matrix-nine-harnesses`](https://github.com/nathansutton/chad/tree/archive/matrix-nine-harnesses/benchmarks/matrix).

---

*Correctness is measured by [`benchmarks/polyglot`](../benchmarks/polyglot/README.md):
215 exercises in six languages, run by the shipped agent on the laptop. Its pass rates are
not quoted here, because its job is telling two builds apart with a paired test, not
ranking chad. Every published run, trajectories included, is in the
[dataset of eval runs](https://huggingface.co/datasets/nathansutton/chad-polyglot-runs).*
