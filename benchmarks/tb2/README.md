# Reproducing chad's Terminal-Bench 2.0 run

*Sonnet on your laptop — and you don't have to take our word for it.*

This directory is the complete, public kit behind the Terminal-Bench 2.0 chart in the
[top-level README](../../README.md): the exact Harbor agent adapter that produced the
runs, the runner script, and the trajectory validator. Nothing here is a cleaned-up
reenactment — `harbor_chad_tb2.py` is the artifact, published.

> **Status: the verified reference run is still in flight.** The maintainer's full
> Ornith-35B run is being completed now; every number in the top-level README is
> provisional until it lands. This kit is published *ahead* of the final number on
> purpose — so that when the number lands, anyone can check it rather than trust it.

## What gets measured

[Terminal-Bench 2.0](https://www.tbench.ai/leaderboard) is the standard benchmark for
CLI coding agents: 89 tasks of real terminal work (build a Cython extension, recover a
WAL'd SQLite database, configure a git webserver, boot a QEMU VM…), each run in its own
Docker container and verified by the *container's end state*, with a hard per-task time
budget. Frontier models sit at the top of the leaderboard; chad isn't competing with
them. The claim under test is narrower and more interesting: **chad + Ornith-35B —
weights served on your own Mac — lands in the neighborhood of Claude Sonnet 4.5's
score.** Sonnet on your laptop. 🗿

## Architecture of a run

The benchmark runs from a Mac, whole: Harbor and the task containers on the machine,
and the model served beside them by `chad --serve` — the same in-process MLX engine
chad ships, warm prefix cache included.

```
your Mac
┌──────────────────────────────────┐      ┌──────────────────────────────┐
│ harbor ──▶ task container        │      │ chad --serve   :8080         │
│            ┌───────────────────┐ │ /v1  │ Ornith-1.0-35B UD-Q2_K_XL    │
│            │ chad (installed   │ │◀────▶│ (in-process MLX engine, warm │
│            │ by setup(), runs  │ │      │  prefix KV; ~13 GB resident) │
│            │ the real shell)   │ │      └──────────────────────────────┘
│            └───────────────────┘ │        reached from containers as
└──────────────────────────────────┘        http://host.docker.internal:8080/v1
```

chad is installed *into* each task container (TB2 tasks are verified by container
end-state, so the agent must act inside it) and generation is remote: the container
runs chad with a remote backend pointed at the server. No weights load in the
container; on Linux chad's install is torch-free and MLX-free (both are Darwin-gated),
so the venv fits TB2's small task images.

> **The 24 GB Mac arm.** Earlier reference sweeps served Ornith from a GPU box via
> llama.cpp (`chad_backend=llama`, [below](#alternative-arm-llamacpp-on-a-gpu-box))
> because a 24 GB Mac couldn't reliably hold the 35B *and* run the benchmark. The
> plan-075 work (Metal memory clamps, catchable-OOM prefill retry, adaptive chunking,
> pressure-aware context governor, and `chad --serve`'s warm-cache serving) exists to
> retire that substitution: the target configuration is the whole benchmark on one
> 24 GB Mac. Until a full 24 GB reference pass is published, treat 24 GB as
> supported-but-being-validated and ≥ 32 GB as the comfortable path.

## Requirements

- An **Apple Silicon Mac** — **≥ 32 GB is comfortable**; **24 GB is the target
  configuration** for the Mac-native arm (see the note above), **Docker** (Docker
  Desktop, VM capped at ~4 GB on a 24 GB box), and **[uv](https://docs.astral.sh/uv/)**.
- **[Harbor](https://harborframework.com)** — Terminal-Bench 2's harness:
  `uv tool install harbor` (the reference runs used **harbor 0.16.1**).
- **`HF_TOKEN` (recommended)** — export a [Hugging Face access
  token](https://huggingface.co/settings/tokens) in the shell that runs `run_tb2.sh`.
  `setup()` pre-fetches the tokenizer into each container's HF cache before chad ever
  runs (so a later load is offline and can't race the Hub); an unauthenticated prefetch
  is more likely to get rate-limited, and a prefetch that fails after 5 retries now
  fails the trial's *setup* (harbor records an env failure and the task can be
  re-run) rather than silently degrading to an online load at chad startup.
- Disk: ~12 GB of model weights, ~1 GB of dataset export, plus tens of GB of Docker
  image cache over a full run. Time: many tasks carry 20–60 minute budgets — a full
  89-task pass is an overnight-to-a-day affair, dominated by serving throughput.

## Step 1 — serve Ornith on the Mac

Serve the exact quant chad ships (`UD-Q2_K_XL`, ~13.4 GB resident) with **chad's own
server** — the same in-process engine the TUI runs, behind an OpenAI-compatible
endpoint:

```bash
CHAD_MODEL=nathansutton/Ornith-1.0-35B-UD-Q2_K_XL-MLX \
  chad --serve --host 0.0.0.0 --port 8080
```

Why chad's server and not `mlx_lm.server`: Ornith's hybrid SSM/attention cache is
non-trimmable, and `mlx_lm.server` cannot reuse its prompt cache across requests — so
every agent step re-prefills the whole growing transcript (tens of seconds per step at
TB2 context sizes; the failure mode that originally pushed our reference runs off the
Mac). `chad --serve` keeps the engine's persistent prefix KV cache warm across an
agent's steps: a warm step prefills only the newly appended tokens (measured: 1.1 s vs
11.7 s on a 7.7k-token transcript, 35B on a 24 GB M4 Pro). `mlx_lm.server` remains a
valid *cold-cache* arm — score it, label it.

Fidelity notes, each load-bearing:

- **Quant.** The Mac arm serves the **shipped 2-bit MLX quant** — the configuration
  the "Sonnet on your laptop" claim is about. Any other quant (a GGUF on a GPU box, a
  higher-bit MLX build) is a *different arm*: score it, report it, but label it.
- **Sampling.** chad sends `temperature` per-request (the runner defaults it to
  **1.0**, Ornith's reported TB2.1 recipe) and does not send top-p/top-k/min-p —
  leave the server's neutral defaults alone.
- **`--host 0.0.0.0`** is what lets the task containers reach the server through
  Docker Desktop's `host.docker.internal`.
- **One task at a time.** The runner drives tasks serially (`--n-concurrent-agents 1`)
  so the single-slot server is never contended and prompt-cache reuse stays honest.

## Step 2 — run

```bash
cd benchmarks/tb2

# Smoke one task first (downloads the dataset on first run):
CHAD_BASE_URL=http://host.docker.internal:8080/v1 ./run_tb2.sh 0 1 fix-git

# The full 89-task pass:
CHAD_BASE_URL=http://host.docker.internal:8080/v1 ./run_tb2.sh
```

`CHAD_BASE_URL` must be reachable **from inside the task containers** —
`host.docker.internal` when the server runs on the Mac itself.

The script reads each task's `[agent] timeout_sec` and gives chad `budget − 30` seconds
so chad's own catchable timeout fires just before harbor's hard SIGKILL — the run stays
faithful to per-task budgets *and* every trial's stdout/trajectory survives the timeout
path. Two optional positional args: `MAXCAP` (cap any single task's chad budget, `0` =
uncapped) and `REPEATS` (harbor `-k`).

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

## Alternative arm: Q6 on a rented 48 GB Mac (EC2 `mac-m4pro.metal`)

The cheapest way to run a **higher-bit all-Apple arm** — the same MLX serving stack,
6-bit weights instead of 2-bit. A 48 GB M4 Pro's ~36 GB Metal working set holds the
~28.5 GB Q6 comfortably alongside a 120k-token window. On AWS that's an EC2
`mac-m4pro.metal` dedicated host; **note the 24-hour minimum allocation** — plan the
day: an n=3 sweep of a task subset fits easily, a full 89-task pass is
overnight-scale per pass.

Setup on the fresh host — the prebuilt quant lives at
[`nathansutton/Ornith-1.0-35B-Q6-MLX`](https://huggingface.co/nathansutton/Ornith-1.0-35B-Q6-MLX)
(~28.5 GB), so there is nothing to build:

```bash
# 0. one-time host deps (EC2 Macs are bare metal — Docker Desktop works; start it
#    and verify `docker info` EARLY, it gates everything else)
brew install uv git && brew install --cask docker
uv tool install harbor

# 1. clone + install
git clone https://github.com/nathansutton/mlxcc && cd mlxcc && uv sync

# 2. serve (downloads the quant on first run; foreground — use tmux/screen so the
#    host survives your ssh dropping)
benchmarks/tb2/serve_q6.sh

# 3. run — smoke one task, then n=3 passes (separate shell)
cd benchmarks/tb2
CHAD_BASE_URL=http://host.docker.internal:8080/v1 ./run_tb2.sh 0 1 fix-git
CHAD_BASE_URL=http://host.docker.internal:8080/v1 ./run_repeated.sh 3
```

To rebuild the quant from the bf16 source instead (verifiable supply chain, ~65 GB
download): `uv run python benchmarks/tb2/quantize_q6.py`, then
`benchmarks/tb2/serve_q6.sh <built-dir>`.

Disk: ~29 GB quant + Docker image cache (+~65 GB bf16 only if rebuilding) — size
the root volume ≥ 200 GB. Per the fidelity notes: this is a **different arm** from
the shipped 2-bit laptop configuration (same class as the llama.cpp Q6_K arm below,
but MLX group-64 affine on Apple silicon, warm-cache serving included) — score it,
label it `q6-mlx-48gb`, don't blend it into laptop numbers.

## Alternative arm: llama.cpp on a GPU box

If your Mac is under 32 GB (ours is — this is the arm our in-flight reference sweep
uses), serve an Ornith-35B GGUF from any machine with the memory and point the kit at
it over your LAN or tailnet:

```bash
hf download bartowski/deepreinforce-ai_Ornith-1.0-35B-GGUF \
  --include "deepreinforce-ai_Ornith-1.0-35B-Q6_K.gguf" --local-dir ~/models/ornith-35b

llama-server \
  --model ~/models/ornith-35b/deepreinforce-ai_Ornith-1.0-35B-Q6_K.gguf \
  --alias ornith-35b-q6k --host 0.0.0.0 --port 8081 \
  -ngl 999 -c 49152 --parallel 1 \
  --top-p 1.0 --top-k 0 --min-p 0 -n 8192

CHAD_BACKEND=llama CHAD_BASE_URL=http://<box-ip>:8081 ./run_tb2.sh
```

On this arm chad speaks llama.cpp's **native `/completion` endpoint with token-id
prompts** (`src/chad/completion_engine.py`) — no chat template runs server-side.
top-p/top-k/min-p must be neutralized in the server flags as shown (chad doesn't send
them), and `--parallel 1` keeps the prompt cache single-slot. The GGUF ships no
tokenizer, so the runner's `CHAD_TOKENIZER` default (the MLX repo — Ornith's vocab is
quant-invariant) is what labels the prompts. It's a different quant on different
silicon: a separate arm, honestly labeled, not the laptop configuration.

## Caveats (read before comparing numbers)

- **The reference run is not finished.** Until the top-level README replaces its
  provisional numbers with the verified run, treat every chart point as directional.
- **An arm = quant × serving stack × throughput.** Score whatever you like, but label
  it. The Mac arm (shipped MLX quant, `mlx_lm.server`) is the claim under test; our
  GPU-box arm exists because of a 24 GB dev machine, and its numbers don't transfer
  quant-for-quant.
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
- The `-m` model label is cosmetic (no weights load from it); the truth of the arm is
  what your server is actually serving. Preflight it: `curl <server>/v1/models` should
  report the model you think you're testing.
