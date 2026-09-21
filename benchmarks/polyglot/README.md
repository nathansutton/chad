# Polyglot: chad's coding eval, reproducible on one Mac

215 hard Exercism exercises in C++, Go, Java, JavaScript, Python and Rust. The agent gets
the problem statement and a stub, works in a real directory with a real toolchain, and
passes only if the exercise's full test suite exits 0. Everything runs on the laptop the
model runs on: no Docker, no server, no grader in the cloud.

The exercises are [Aider's polyglot benchmark](https://github.com/Aider-AI/polyglot-benchmark),
pinned to one upstream commit and fetched rather than vendored. The numbers here are
**not comparable to Aider's leaderboard**: that protocol is two attempts at a whole-file
edit with test output fed back once, and this one is an agent loop that may run the
tests as often as its step and time budget allow. Same problems, different exam.

## Why this exists

An eval is only useful if it can tell two versions of the agent apart. chad's hand-built
task tiers cannot any more — the shipped model passes all of them — and a benchmark that
is always 100% gives zero information about a change. These exercises were selected
upstream because strong models fail them, and they cost minutes each instead of the
better part of an hour, so a laptop can afford the repetitions that statistics need.

## Run it

```sh
uv run python benchmarks/polyglot/prepare.py     # once: exercises, jars, jest, crates
uv run python benchmarks/polyglot/gold.py        # once: prove the toolchains, no model
uv run python benchmarks/polyglot/run.py --label baseline --reps 3
python benchmarks/polyglot/stats.py score benchmarks/polyglot/_runs/baseline/trials.jsonl
```

Each trial writes an ATIF trajectory to `_runs/<label>/trajectories/<language>/`, and
rewrites it while the trial runs. To see where a slow trial is spending its time:

```sh
python benchmarks/polyglot/trace.py benchmarks/polyglot/_runs/baseline/trajectories/go/book-store.rep1.json
```

Toolchains: `python3`, `go`, `cargo`, `node`+`npm`, a JDK (17 or newer), `cmake` and a
C++ compiler. `prepare.py` names any that are missing; their tasks are skipped, not failed.

A block loads the weights once and is resumable: re-run the same command after a crash
or a closed lid and it continues from the last recorded trial. Budget four to five
minutes a task on a 24 GB M4 Pro: about 16 hours for one pass over all 215.

## What keeps a score honest

- **The gold gate.** Before any model runs, every exercise is verified twice: its
  untouched stub must fail and upstream's reference solution must pass. 215 of 225 clear
  both and are listed in `manifest.json`; the other 10 are excluded with the reason (five
  are refactoring exercises whose stub already passes, five have reference solutions that
  need Boost, RxJava, org.json or a specific ICU locale). Scores share that denominator.
- **No answer in the workspace.** `.meta/` (reference solution), `.approaches/` (prose
  walkthroughs) and the hints file never reach the trial directory or the prompt.
- **Tests belong to the harness.** Test files, build files and `run-tests.sh` are
  restored from upstream before grading, so editing a test or the script earns nothing.
- **The whole suite counts.** Upstream ships most tests disabled for Exercism's teaching
  flow; all of them are switched on, as Aider's harness does.
- **Exit status decides.** No output parsing, so a test runner's wording cannot move a score.
- **Offline trials.** Dependencies are fetched up front and pinned by commit, lockfile or
  SHA-256; nothing a trial runs reaches the network.
- **It is the shipped agent.** Shipped model, shipped sampler preset, RAM-aware context
  limit, default step budget. The one cap the benchmark adds is wall clock per trial
  (`--wall-cap`, 1200 s), recorded per row as `capped`.

## Comparing two versions

Pass rates of two runs differ mostly because tasks differ in hardness, which says nothing
about the change. `stats.py compare` pairs by task and runs an exact sign test on the
tasks whose pass rate moved (with one rep per arm, that is McNemar's exact test), so
significance depends on how many tasks flipped, not how many were run. It needs at least
six one-directional flips to reach p < 0.05.

To make flips cheap, pin a **pool** from a baseline with two or more reps — the tasks
the baseline passes sometimes but not always — before looking at the arm under test:

```sh
python benchmarks/polyglot/stats.py pool _runs/baseline/trials.jsonl > results/pool.txt
CHAD_DISABLE=env_manifest uv run python benchmarks/polyglot/run.py \
    --label no-manifest --tasks-file benchmarks/polyglot/results/pool.txt --reps 3
python benchmarks/polyglot/stats.py compare _runs/baseline/trials.jsonl _runs/no-manifest/trials.jsonl
```

The pool authorizes a change; the full set only vetoes one, by showing it broke tasks
that used to be safe.

## Files

| file | job |
|---|---|
| `catalog.py` | pinned upstream, the task list, `manifest.json` loader |
| `workspace.py` | leak-free trial directories, `run-tests.sh`, `verify()` |
| `prepare.py` | one-time fetch of everything trials need from the network |
| `gold.py` | the gold gate; writes `manifest.json` |
| `run.py` | a block of trials: one model load, one row per trial |
| `stats.py` | score, pool, paired compare |
| `trace.py` | one trial's ATIF trajectory as a step table: tokens, think, seconds, why each step ended — readable while the trial is still running |
| `results/` | committed runs: `meta.json` + `trials.jsonl` per label |

Each row records pass/fail, wall clock, steps, generated / thinking / prefilled tokens,
peak context, which levers fired, and the machine's thermal state. `meta.json` records
the chad version and commit, the model, the context limit and every `CHAD_*` variable.
