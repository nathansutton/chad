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
and the model served beside them by `mlx_lm.server` — the same MLX stack chad ships.

```
your Mac
┌──────────────────────────────────┐      ┌──────────────────────────────┐
│ harbor ──▶ task container        │      │ mlx_lm.server  :8080         │
│            ┌───────────────────┐ │ /v1  │ Ornith-1.0-35B UD-Q2_K_XL    │
│            │ chad (installed   │ │◀────▶│ (MLX — the exact quant chad  │
│            │ by setup(), runs  │ │      │  ships; ~12 GB resident)     │
│            │ the real shell)   │ │      └──────────────────────────────┘
│            └───────────────────┘ │        reached from containers as
└──────────────────────────────────┘        http://host.docker.internal:8080/v1
```

chad is installed *into* each task container (TB2 tasks are verified by container
end-state, so the agent must act inside it) and generation is remote: the container
runs chad with a remote backend pointed at the server. No weights load in the
container; on Linux chad's install is torch-free and MLX-free (both are Darwin-gated),
so the venv fits TB2's small task images.

> **Why our own reference run substitutes a GPU box.** The maintainer's dev Mac has
> 24 GB — not enough to hold the 35B *and* run the benchmark — so our in-flight
> reference sweep serves Ornith from a GPU server that happened to be on the shelf,
> via llama.cpp (`chad_backend=llama`, [below](#alternative-arm-llamacpp-on-a-gpu-box)).
> That's a memory limit, not a design choice: on a ≥32 GB Mac the whole thing runs on
> the laptop, and that Mac-native arm is the configuration the branding claims.

## Requirements

- An **Apple Silicon Mac with ≥ 32 GB** (the 35B tier — same floor as chad itself),
  **Docker** (Docker Desktop), and **[uv](https://docs.astral.sh/uv/)**.
- **[Harbor](https://harborframework.com)** — Terminal-Bench 2's harness:
  `uv tool install harbor` (the reference runs used **harbor 0.16.1**).
- Disk: ~12 GB of model weights, ~1 GB of dataset export, plus tens of GB of Docker
  image cache over a full run. Time: many tasks carry 20–60 minute budgets — a full
  89-task pass is an overnight-to-a-day affair, dominated by serving throughput.

## Step 1 — serve Ornith on the Mac

Serve the exact quant chad ships (`UD-Q2_K_XL`, ~12 GB resident) with `mlx_lm.server`:

```bash
uvx --from mlx-lm mlx_lm.server \
  --model nathansutton/Ornith-1.0-35B-UD-Q2_K_XL-MLX \
  --host 0.0.0.0 --port 8080
```

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
