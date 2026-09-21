# Contributing to chad

chad is a local, single-user, Apple-Silicon coding agent. It's a small project with a
sharp design constraint (prefill is the bill, the KV cache stays warm), so here's the
honest map of what lands easily and what needs a conversation first.

Working here with an agent? Start at [`AGENTS.md`](AGENTS.md); the per-module map is in
[`docs/architecture.md#architecture-map`](docs/architecture.md#architecture-map).

## What lands easily

Docs fixes, tests, bug fixes that come with a failing-test repro, portability and
tooling improvements. The gate is fast and needs **no model weights**:

```bash
make gate
```

That runs four targets in order, and CI (`.github/workflows/tests.yml`) runs the same ones:

- `make lint` runs `ruff check` over `src`, `tests` and `benchmarks`.
- `make typecheck` runs `mypy` over `src/chad`.
- `make slop` runs the vendored [anti-slop](https://github.com/TinyFrontier/anti-slop-py)
  linter (`tools/anti_slop`, stdlib-only, needs Python 3.12 — `uv` fetches one) over the
  same three trees. It rejects the escape hatches ruff and mypy permit by construction:
  `Any`/`object` contracts, `dict[str, Any]`, string-name `getattr`, `mock.patch`, and any
  `cast` or `# type: ignore[code]` without a `# SAFETY: <invariant>` comment. Findings
  that predate the linter are recorded in `.anti-slop-baseline.json` and resurface when
  their line is edited; `make slop-review` shows only what your branch added, with the
  recipe for each.
- `make test` runs `pytest -q`, which loads no model and finishes in seconds.

A green `make test` alone still fails the build if `ruff`, `mypy` or anti-slop is unhappy,
so run the whole gate before opening a PR.

### Running the model-backed tests

A few tests in `tests/test_engine.py` load a real model and compare its output byte for
byte. They skip unless you set `CHAD_MODEL_TESTS=1`, and the first run downloads a 0.5B
proxy model from Hugging Face:

```bash
CHAD_MODEL_TESTS=1 uv run pytest -q tests/test_engine.py
```

The hybrid-cache tests also need `CHAD_TEST_HYBRID_MODEL` set to a local qwen3_5 model
directory, and skip without it. Their strict byte-equality checks need unquantized (bf16)
weights.

## What needs a conversation first

Anything that changes **model-visible behavior**: prompts, tool schemas, guardrails, the
engine, compaction. These are validated on the maintainer's **private eval rig** (the
core/hard/realworld/brutal tiers that self-skip here for lack of weights), which a PR can't
run. I can't merge a behavior change on the unit tests alone; I have to take it to the rig
myself. So please **open an issue and describe the change before building it**, or your PR
may stall waiting on an eval pass you can't see. That is how a RAM-bound local model gets
kept honest.

## Dev setup

```bash
uv sync                # install deps + the `chad` console script (one time)
uv run chad            # launch the full-screen TUI
uv run chad "do X"     # one-shot headless task, then exit
uv run chad -c         # resume this directory's saved conversation
uv run pytest -q       # fast unit gate
```

Python is 3.11+; dependency and venv management is [uv](https://docs.astral.sh/uv/).

## High-risk zones

Two areas corrupt more than the line you touched, so lean on the existing tests:

- **`engine.py` and `compaction.py`.** The shipped model's hybrid SSM/attention cache is
  **non-trimmable**: any change to the prefix forces a full re-prefill, and a bug here
  corrupts every later turn. Run `test_engine.py` / `test_compaction.py`.
- **The `run_turn` loop in `agent.py`.** The heart of the agent, guarded by
  `test_agent_guards.py`, with tool/edit behavior in `test_tools.py` / `test_edit.py`.

## Style

ruff and mypy are the law. Match the surrounding comment density and naming, and write code
that reads like the code already there.

`build/` and `*.egg-info` are regenerable artifacts; delete them before a repo-wide grep or
a local wheel build.
