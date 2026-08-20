# Reproducing chad's Terminal-Bench 2.1 run

*Frontier scores at laptop cost — and you don't have to take our word for it.*

This directory is the complete, public kit behind the Terminal-Bench 2.1 chart in the
[top-level README](../../README.md): the exact Harbor agent adapter that produced the
runs, the runner script, and the trajectory validator. Nothing here is a cleaned-up
reenactment — `harbor_chad_tb2.py` is the artifact, published.

> **Status: chad 2.0.0 ships no scored claim.** 2.0.0 replaced the default model
> (Ornith-35B → Qwen3.8-27B), which invalidates any score measured on the old one, and
> the new default has not been scored on Terminal-Bench 2.1 yet. The recipe below is the
> **Ornith-35B arm** — the one the retired 57% (51/89, k=1, self-run) came from, kept
> because it is the artifact that produced a published number and it still runs against
> those weights. To score the *shipped* model, keep the adapter and the runner and swap
> the server (see [Serving the 2.0.0 default](#serving-the-200-default-instead)).

## What gets measured

[Terminal-Bench 2.1](https://www.tbench.ai/leaderboard) is the standard benchmark for
CLI coding agents: 89 tasks of real terminal work (build a Cython extension, recover a
WAL'd SQLite database, configure a git webserver, boot a QEMU VM…), each run in its own
Docker container and verified by the *container's end state*, with a hard per-task time
budget. Frontier models sit at the top of the leaderboard; chad isn't competing with
them. The claim under test is narrower and more interesting: **weights you serve
yourself land in the neighborhood of a frontier model's score, for electricity instead
of $130–$2,000 of API fees.** The retired Ornith-35B arm reached 57% that way; whether
the 27B does is exactly what this kit is for. 🗿

## Architecture of a run

chad is installed *into* each task container (TB2 tasks are verified by container
end-state, so the agent must act inside it) and generation is remote: the in-container
chad runs the `--backend llama` engine pointed at a server you provide. No weights load in
the container; on Linux chad's install is torch-free and MLX-free (both are Darwin-gated),
so the venv fits TB2's small task images.

```
your machine
┌──────────────────────────────────┐      ┌──────────────────────────────┐
│ harbor ──▶ task container        │      │ a /completion server  :8081  │
│            ┌───────────────────┐ │ /comp│   `chad serve`  (MLX, the    │
│            │ chad (installed   │ │◀────▶│    shipped weights), or      │
│            │ by setup(), runs  │ │      │   `llama-server` (a GGUF)    │
│            │ the real shell)   │ │      └──────────────────────────────┘
│            └───────────────────┘ │        reached from containers as
└──────────────────────────────────┘        http://host.docker.internal:8081
```

**Why the raw `/completion` protocol.** chad the product runs an MLX quant *in-process* on
Apple Silicon — that's the TUI you use day to day. But the benchmark harness runs chad
inside a Linux container, where MLX can't load, so generation must be remote. llama.cpp's
native `/completion` endpoint is the right shape for it: it accepts chad's exact rendered
prompt as **token ids** (no double chat-template, no re-tokenization drift), reuses the
longest common prefix across requests with `cache_prompt` (a warm step prefills only the
appended tokens, not the whole transcript), and passes `<think>` back **verbatim** — the
last point rules out `mlx_lm.server`, which strips the reasoning block and is unusable for
a reasoning model.

Two servers speak it. **`chad serve`** backs that same protocol with the local MLX engine
and its real prefix cache, so the container drives the exact artifact people install —
that is the arm to prefer for scoring the shipped model. **`llama-server`** holds a GGUF
and is what the retired Ornith run used; it is also the only option if the weights are too
big for the Mac running the harness. A GGUF served by llama.cpp and chad's in-process MLX
quant are *different arms* — an honest caveat, called out again below.

## Requirements

- **A machine that can serve the weights.** For the shipped model that is any 24 GB+
  Apple Silicon Mac running `chad serve` (~12 GB resident). For the retired Ornith arm it
  is a Mac with enough unified memory (Q6_K is ~28.5 GB resident → ≥ 48 GB comfortable;
  smaller quants fit less RAM) or any GPU box with the VRAM. The machine running the
  benchmark needs **Docker** (Docker Desktop on a Mac) and
  **[uv](https://docs.astral.sh/uv/)**. Server and harness can be the same machine or two
  boxes on a LAN/tailnet — but note that on one machine the task containers compete with
  decode, and TB2 budgets are wall-clock.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — only for the GGUF arm:
  `llama-server` (Metal build on a Mac, CUDA/Vulkan on a GPU box). `chad serve` needs
  nothing beyond chad itself.
- **[Harbor](https://harborframework.com)** — Terminal-Bench 2's harness:
  `uv tool install harbor` (the reference runs used **harbor 0.16.1**).
- **`HF_TOKEN` (recommended)** — export a [Hugging Face access
  token](https://huggingface.co/settings/tokens) in the shell that runs `run_tb2.sh`.
  `setup()` pre-fetches the tokenizer into each container's HF cache before chad ever
  runs (so a later load is offline and can't race the Hub); an unauthenticated prefetch
  is more likely to get rate-limited, and a prefetch that fails after 5 retries now
  fails the trial's *setup* (harbor records an env failure and the task can be
  re-run) rather than silently degrading to an online load at chad startup.
- Disk: the weights (~12 GB for the shipped MLX model, ~29 GB for the Q6_K GGUF), ~1 GB
  of dataset export, plus tens of GB of Docker
  image cache over a full run. Time: many tasks carry 20–60 minute budgets — a full
  89-task pass is an overnight-to-a-day affair, dominated by serving throughput.

## Step 1 — serve the model

Two servers work, because the in-container chad only needs something that speaks
llama.cpp's raw `/completion` with token-id prompts. Pick the arm you mean to score.

### Serving the 2.0.0 default instead

`chad serve` exposes *this Mac's* in-process MLX engine — the shipped Qwen3.8-27B, its
real prefix cache, `<think>` passed back verbatim — over that same protocol, so the
container drives the model people actually install rather than a different quantization
of it:

```bash
uv run chad serve --host 0.0.0.0 --port 8081     # on the Mac holding the weights
```

Then run Step 2 unchanged: `CHAD_BASE_URL` points at it the same way. Two things to know
before trusting a number from this arm — the engine holds one KV cache, so tasks must stay
serial (`--n-concurrent-agents 1`, which the runner already does), and on a laptop the task
containers compete with decode, so a wall-clock-budgeted task can fail on time rather than
on capability. Pilot a handful of tasks before committing to an overnight sweep. Full
details in the [configuration
reference](../../docs/configuration.md#serving-the-local-model-to-a-container-chad-serve).

### Serving Ornith with llama.cpp (the retired arm, reproduced below)

Download an Ornith-35B GGUF and serve it with `llama-server`, single-slot, with the
sampling filters neutralized (by default chad sends only `temperature`; keep the rest
neutral so the recipe is faithful):

```bash
hf download bartowski/deepreinforce-ai_Ornith-1.0-35B-GGUF \
  --include "deepreinforce-ai_Ornith-1.0-35B-Q6_K.gguf" --local-dir ~/models/ornith-35b

llama-server \
  --model ~/models/ornith-35b/deepreinforce-ai_Ornith-1.0-35B-Q6_K.gguf \
  --alias ornith-35b-q6k --host 0.0.0.0 --port 8081 \
  -ngl 999 -c 49152 --parallel 1 \
  --top-p 1.0 --top-k 0 --min-p 0 -n 8192
```

Fidelity notes, each load-bearing:

- **Quant = arm.** A `Q6_K` GGUF, a 2-bit GGUF, and an MLX build are *different arms* on
  *different serving stacks*: score whichever you like, but label it and don't blend it
  into another arm's numbers. The in-flight reference run serves `Q6_K`.
- **Sampling.** chad sends `temperature` per-request (the runner defaults it to **1.0**,
  Ornith's reported TB2.1 recipe) plus `min_p`/`top_p` when armed (`chad_min_p`/
  `chad_top_p`); it never sends top-k. `run_tb21_submit.sh` arms **`min_p 0.05`** by
  default (`CHAD_TB2_MINP`): temp stays 1.0 but the sub-noise-floor quantization
  tail — a source of the garbled-tool-call/foreign-dialect derailments the bf16
  reference recipe never sampled — is cut. `CHAD_TB2_MINP=0` restores the bare
  recipe. Neutralize the server flags (`--top-p 1.0 --top-k 0 --min-p 0`) so
  per-request knobs are the only sampler configuration.
- **`--host 0.0.0.0`** is what lets the task containers reach the server through Docker
  Desktop's `host.docker.internal` (or use the box's LAN/tailnet IP for a separate host).
- **`--parallel 1` + one task at a time.** The runner drives tasks serially
  (`--n-concurrent-agents 1`) so the single prompt-cache slot is never contended and
  prefix reuse stays honest.
- **Tokenizer.** The GGUF ships no tokenizer, so chad loads one client-side; the runner's
  `CHAD_TOKENIZER` default is the MLX repo (Ornith's vocab is quant-invariant, so it's
  right for every Ornith-35B GGUF).

## Step 2 — run

```bash
cd benchmarks/tb2

# Smoke one task first (downloads the dataset on first run):
CHAD_BASE_URL=http://host.docker.internal:8081 ./run_tb2.sh 0 1 fix-git

# The full 89-task pass:
CHAD_BASE_URL=http://host.docker.internal:8081 ./run_tb2.sh

# Terminal-Bench 2.1 (same 89 tasks with upstream task fixes; the dataset Ornith's
# reported recipe targets) — select the dataset with TB2_DATASET:
TB2_DATASET=terminal-bench/terminal-bench-2-1 \
  CHAD_BASE_URL=http://host.docker.internal:8081 ./run_tb2.sh
```

`CHAD_BASE_URL` is the **bare llama-server origin** (no `/v1`), reachable **from inside
the task containers** — `host.docker.internal` when the server runs on the same machine,
a LAN/tailnet IP for a separate box.

Deadline awareness is on by default: the adapter hands chad a wall budget
(`chad_timeout_sec − chad_deadline_margin_s`, margin 60s) via `CHAD_TURN_BUDGET_S`, so the
runaway-turn governor arms on every trial and a one-shot wrap-up nudge fires in the
final stretch ("land your best answer now") instead of letting the SIGKILL eat
half-applied work. `--ak chad_deadline_margin_s=0` disables the whole path for A/Bs.
Two related knobs ship **off** and should stay off for benchmark runs:
`chad_review_pass` (early-finish fresh-context re-verify — burned +140–584s per
already-correct task for zero flips in its gate) and `chad_think_ceiling` (close-and-
continue think salvage — FAILED its eval gate with 2.5× think inflation when armed).

The script reads each task's `[agent] timeout_sec` and gives chad `budget − 30` seconds
so chad's own catchable timeout fires just before harbor's hard SIGKILL — the run stays
faithful to per-task budgets *and* every trial's stdout/trajectory survives the timeout
path. Two optional positional args: `MAXCAP` (cap any single task's chad budget, `0` =
uncapped) and `REPEATS` (harbor `-k`). For a submission-grade multi-trial run use
`./run_tb21_submit.sh` (leaderboard-compliant: no timeout multipliers, public Hub
upload, `-k` defaulting to the leaderboard's 5-trial minimum) — and never publish a
single sample as the number.

Results land in `tb2_full_<stamp>.tsv` (one row per task: budget, cap, reward,
wall-clock) with a pass-rate tally at the end. Full trial artifacts — verifier output,
chad's in-container stdout, the prefill trace, and an ATIF `trajectory.json` per trial
— land under `jobs/` where harbor puts them.

## Step 3 — validate trajectories (leaderboard hygiene)

The leaderboard requires an [ATIF](https://www.tbench.ai) trajectory for every passing
trial. chad emits one natively (`src/chad/atif.py`); check a finished job against
harbor's own schema:

```bash
"$(uv tool dir)/harbor/bin/python" validate_atif.py jobs/<job-dir>/
```

Exit 0 means the bundle is submittable; it fails loudly if any passing trial lacks a
valid trajectory.

## Building the GGUF yourself (optional, verifiable supply chain)

The `Q6_K` GGUF above is a community conversion. To build a quant from the bf16 source
instead, convert with llama.cpp's `llama-quantize` (see its docs). Note the arm rule:
a GGUF served by llama.cpp and chad's shipped in-process MLX quant are separate serving
stacks — numbers don't transfer between them.

## Caveats (read before comparing numbers)

- **The step-by-step recipe is the retired arm.** It reproduces the Ornith-35B run, not
  what `uvx chad-code` installs today. 2.0.0's default is unscored; don't quote the 57% as
  a number about the shipped model. To score the shipped model, keep the adapter and the
  runner and serve with [`chad serve`](#serving-the-200-default-instead).
- **An arm = model × quant × serving stack × throughput.** Score whatever you like, but
  label it. A GGUF via llama.cpp and chad's in-process MLX engine are different artifacts;
  numbers don't transfer quant-for-quant, which is the whole reason `chad serve` exists.
- **Pass-rate is throughput-sensitive.** TB2 budgets are wall-clock, so the score
  measures *the model at your serving speed*, not the model in the abstract. In our
  interim runs a substantial share of failures were **timeouts with correct work in
  progress**, not wrong answers — slower serving lowers the score without the model
  getting any dumber. Report your hardware alongside any number.
- **Temperature 1.0 means variance.** Single runs (`REPEATS=1`) bounce by a few points;
  the leaderboard convention (and ours) is to prefer multi-trial means (`-k`).
- **Install fidelity.** The adapter uploads your working tree and installs it with
  `uv sync --frozen`, so the run measures *your checkout at your lockfile* — that's the
  point (it's how you verify us), but it also means a dirty tree is what gets measured.
- The `-m` model label is cosmetic (no weights load from it); the truth of the arm is what
  your server is actually serving. Preflight it: `curl <server>/health` should answer —
  against `chad serve` that also reports `n_ctx` and the live `safe_ctx`, which is worth a
  look before an overnight run — and against `llama-server`, check the `--alias` you set
  names the model you think you're testing.
