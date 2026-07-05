# Contributing to chad

chad is a local, single-user, Apple-Silicon coding agent. It's a small project with a
sharp design constraint (prefill is the bill, the KV cache stays warm), so here's the
honest map of what lands easily and what needs a conversation first.

## What lands easily

Docs fixes, tests, bug fixes that come with a failing-test repro, portability and
tooling improvements. The gate is fast and needs **no model weights**:

```bash
uv run pytest -q            # unit gate — loads no model, runs in seconds
uv run ruff check src tests # lint
uv run mypy src/chad        # type check
```

All three run in CI (`.github/workflows/tests.yml`) — a green `pytest` alone still fails
the build if `ruff` or `mypy` is unhappy, so run all three before opening a PR.

## What needs a conversation first

Anything that changes **model-visible behavior** — prompts, tool schemas, guardrails,
the engine, compaction. Be warned up front: these are validated on the maintainer's
**private eval rig** (the core/hard/realworld/brutal tiers that self-skip here for lack
of weights), which a PR can't run. That means I can't merge a behavior change on the
unit tests alone — I have to take it to the rig myself. So please **open an issue and
describe the change before building it**, or your PR may stall waiting on an eval pass
you can't see. Not a brush-off — just how a RAM-bound local model gets kept honest.

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

- **`engine.py` and `compaction.py`** — Ornith's hybrid SSM/attention cache is
  **non-trimmable**: any change to the prefix forces a full re-prefill, and a bug here
  corrupts every later turn. Run `test_engine.py` / `test_compaction.py`.
- **the `run_turn` loop in `agent.py`** — the heart of the agent; guarded by
  `test_agent_guards.py`, with tool/edit behavior in `test_tools.py` / `test_edit.py`.

## Style

ruff and mypy are the law. Match the surrounding comment density and naming — write code
that reads like the code already there.
