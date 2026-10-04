# Contributing to chad

chad is a local, single-user, Apple-Silicon coding agent with one sharp constraint: prefill
is the bill, and the KV cache stays warm. This is the map of what lands easily and what
needs a conversation first. Working here with an agent? Start at [`AGENTS.md`](AGENTS.md),
which also holds the high-risk zones, the dependency-pin rule and the code conventions.

## What lands easily

Docs fixes, tests, bug fixes with a failing-test repro, portability and tooling. The gate
is fast and loads no model weights:

```bash
make gate    # lint, typecheck, anti-slop, test — the same four targets CI runs
```

A green `make test` alone still fails CI if ruff, mypy or anti-slop is unhappy, so run the
whole gate before opening a PR. Run `make typecheck` on macOS: on Linux the mlx symbols
type as `Any` and the check passes vacuously over the engine modules it should guard.

A few engine tests load a real model and compare output byte for byte. They skip unless
`CHAD_MODEL_TESTS=1` is set; the first run downloads a 0.5B proxy model:

```bash
CHAD_MODEL_TESTS=1 uv run pytest -q tests/test_engine.py
```

The hybrid-cache tests also need `CHAD_TEST_HYBRID_MODEL` pointing at a local bf16
qwen3_5 model directory, and skip without it.

## What needs a conversation first

Anything that changes **model-visible behavior**: prompts, tool schemas, guardrails, the
engine, compaction. Unit tests cannot say whether such a change helps, so it arrives with a
measurement from [`benchmarks/polyglot`](benchmarks/polyglot/README.md), which runs on
your own Mac: pin a pool of tasks a baseline passes only sometimes, run the pool on `main`
and on your branch, and paste the `stats.py compare` output into the PR. **Open an issue
describing the change before building it.** A paired run costs a night, and it is worth
agreeing on what it should show first.

## Dev setup

```bash
uv sync                # deps + the `chad` console script (one time)
uv run chad            # the TUI
uv run chad "do X"     # one-shot headless task
uv run chad -c         # resume this directory's last conversation
uv run pytest -q       # the fast unit gate
```

Python 3.11+, managed by [uv](https://docs.astral.sh/uv/). `build/` and `*.egg-info` are
regenerable; delete them before a repo-wide grep or a local wheel build.
