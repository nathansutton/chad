# `benchmarks/tb2/` — the containerised end-to-end harness (maintainer tooling)

This directory is the adapter that runs chad inside a Linux task container against a
remote model server, for end-to-end agent evaluation on a
[Harbor](https://github.com/laude-institute/harbor)-style task set. It is kept as a
**development tool**, not as a claim: chad's premise is a 24 GB Apple Silicon laptop with
the model in-process, and a run that serves the weights from another machine into a
container measures a different system. **No number from this kit appears in chad's
documentation**, and none should be added — see the repo's release checklist.

What it is useful for is *paired* comparison: two builds of chad, same tasks, same server,
same seed, where the absolute score is irrelevant and only the delta between arms matters.
That is how the 2.0.0 tool-surface and lever decisions were checked
([Design & internals](../../docs/design.md#why-the-tool-surface-is-five-tools)).

## How a run is shaped

chad is installed *into* each task container (tasks are verified by the container's end
state, so the agent must act inside it) and generation is remote: the in-container chad
runs the `--backend llama` engine pointed at a `/completion` server you provide. No weights
load in the container; on Linux chad's install is torch-free and MLX-free (both are
Darwin-gated), so the venv fits small task images.

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

**Why the raw `/completion` protocol.** llama.cpp's native endpoint accepts chad's exact
rendered prompt as **token ids** (no double chat-template, no re-tokenization drift), reuses
the longest common prefix across requests with `cache_prompt` (a warm step prefills only
the appended tokens), and passes `<think>` back **verbatim** — which rules out
`mlx_lm.server`, which strips the reasoning block and is unusable for a reasoning model.
Two servers speak it: [`chad serve`](../../docs/configuration.md#serving-the-local-model-to-a-container-chad-serve)
backs it with the local MLX engine and the shipped weights, and `llama-server` backs it with
any GGUF from any box that can hold one.

## Files

| File | What it is |
|---|---|
| `harbor_chad_tb2.py` | the Harbor agent adapter: installs chad into the container, runs the task, collects the trajectory |
| `run_tb2.sh` / `run_tb21_submit.sh` | runner scripts (the latter writes ATIF trajectories for every trial) |
| `validate_atif.py` | checks a run's trajectories against the ATIF schema |

## Running it

1. Serve a model on `:8081` — `uv run chad serve --host 0.0.0.0 --port 8081` on a Mac, or
   `llama-server --model <gguf> --host 0.0.0.0 --port 8081 --ctx-size 65536 --parallel 1
   --cache-prompt` anywhere.
2. Smoke one task first, then the set — `run_tb2.sh` reads its dataset and task selection
   from the environment; the header of each script documents the variables.
3. `validate_atif.py <run dir>` before reading anything into a result.

## Caveats

- **Throughput is part of the score.** Tasks carry wall-clock budgets, so the serving
  speed of the host decides some tasks on time rather than capability. That is one reason
  absolute numbers from this rig describe the rig, not the product.
- **The warm-prefix checkpoint and cache quarantine are off** under `--backend llama` — the
  KV lives in the server — so the in-container agent is not the in-process agent.
- **Sampling is not deterministic** across servers or versions; run arms paired, same
  server, same day, and compare deltas.
