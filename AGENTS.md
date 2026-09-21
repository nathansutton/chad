# AGENTS.md

chad is a local, single-user, Apple-Silicon coding agent: one MLX engine, one five-tool
surface, one agent loop. Why it is built this way is in `docs/design.md`; what each module
does, and which tests guard it, is the table in
[`docs/architecture.md#architecture-map`](docs/architecture.md#architecture-map). Read those before
proposing a redesign — this file does not repeat them.

## Gate

Run `make gate` before you call anything done. It runs four targets in order, and CI runs
the same four:

- `make lint` — `uv run ruff check src tests benchmarks`
- `make typecheck` — `uv run mypy src/chad`
- `make slop` — the vendored anti-slop linter (`tools/anti_slop`) over `src`, `tests`
  and `benchmarks`; stdlib-only, runs on its own 3.12 interpreter
- `make test` — `uv run pytest -q`, which loads no model and finishes in seconds

A green `make test` alone still fails the build if ruff, mypy or anti-slop is unhappy, so
run the whole gate. Run `make typecheck` **on macOS**: on Linux mlx is not installed,
`ignore_missing_imports` types every mlx symbol as `Any`, and the check passes vacuously
over exactly the engine and kernel modules it should guard.

The suite is model-free by default. `CHAD_MODEL_TESTS=1 uv run pytest -q
tests/test_engine.py` opts into the 0.5B proxy-model tier (first run downloads it); the
hybrid-cache tests additionally need `CHAD_TEST_HYBRID_MODEL` set to a local bf16 qwen3_5
model dir and skip without it.

## High-risk zones

Two areas corrupt more than the line you touched, so lean on the existing tests:

- **`engine.py` and `compaction.py`.** The shipped model's hybrid SSM/attention cache is
  **non-trimmable**: any change to the prefix forces a full re-prefill, and a bug here
  corrupts every later turn. Run `tests/test_engine.py` and `tests/test_compaction.py`.
- **The `run_turn` loop in `agent.py`.** The heart of the agent, guarded by
  `tests/test_agent_guards.py` and `tests/test_agent_e2e.py`, with tool/edit behavior in
  `tests/test_tools.py` and `tests/test_edit.py`.

## Model-visible changes need a conversation

Prompts, tool schemas, guardrails, the engine and compaction change **model-visible
behavior**. Those are validated on the maintainer's private eval rig, which a PR cannot
run and which self-skips here for lack of weights, so the unit tests alone cannot carry
them. Open an issue describing the change before building it, or the work stalls waiting
on an eval pass you cannot see.

## Dependency pins are load-bearing

The `mlx`, `mlx-lm`, `transformers` and `mcp` bounds in `pyproject.toml` are hand-gated
against this kernel stack and its loader, not resolver defaults — one of them excludes a
release that generates fluent garbage on these kernels, another a break that fails
silently through `getattr` defaults. Never widen a pin to satisfy a resolver: read the
comment above the bound you want to move, re-gate that release, then move it.

## Conventions

- ruff, mypy and anti-slop are the law. Match the surrounding comment density and naming,
  and write code that reads like the code already there.
- anti-slop bans the escape hatches a checker error tempts you into: no `Any`/`object`
  parameters or returns, no `dict[str, Any]`, no `getattr`/`setattr` by string, no
  `mock.patch`, and every `cast` or `# type: ignore[code]` carries a `# SAFETY: <why>`
  comment on its line or directly above. Name the contract instead: parse at the I/O
  boundary, declare the type, inject the seam. `make slop-review` reads back only the
  findings on lines your change touched, each with the recipe that replaces it;
  `uv run --python 3.12 --no-project python tools/anti_slop --explain <rule>` gives the
  reasoning. Pre-existing findings sit in `.anti-slop-baseline.json` and come back the
  moment their line is edited — fix them then. Never regenerate the baseline, add an
  `# anti-slop: ignore[...]`, or set a rule to `off` to make a new finding go away.
- Comments carry behavioural rationale only — no plan numbers, kata refs or benchmark
  references in code.
- The version lives in **both** `pyproject.toml` and `src/chad/__init__.py` and the two
  must match; see `RELEASING.md`.
- `plans/` at the repo root is where handoff plans live and is plan mode's only writable
  path; it is the user's to commit or ignore.
- `build/` and `*.egg-info` are regenerable artifacts; delete them before a repo-wide grep
  or a local wheel build.
