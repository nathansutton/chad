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
cd benchmarks/polyglot
python stats.py pool _runs/baseline/trials.jsonl > _runs/baseline/pool.txt
CHAD_DISABLE=env_manifest uv run python run.py \
    --label no-manifest --tasks-file _runs/baseline/pool.txt --reps 3
python stats.py compare _runs/baseline/trials.jsonl _runs/no-manifest/trials.jsonl
```

The pool authorizes a change; the full set only vetoes one, by showing it broke tasks
that used to be safe.

## Other harnesses

`--harness` runs the same tasks, prompt, wall cap and verification under another coding
agent's own command line: an entry of `harnesses.py` (pi, opencode, dsh, goose,
mini-swe-agent, crush, cline, codex), or `chad-llama`, chad's own CLI on the same server
as them, which is the arm every foreign one is paired against. They all talk to one
llama-server, started once for the whole llama phase:

```sh
uv run python benchmarks/polyglot/server.py        # the matrix's GGUF, -c 32768, --jinja; Ctrl-C stops it
uv run python benchmarks/polyglot/run.py --label h3-pi --harness pi \
    --tasks-file benchmarks/polyglot/subsets/harness-3.txt --reps 3
```

One engine at a time, enforced: an in-process block refuses while a llama-server is up,
and a block refuses while another block runs.

Each CLI block puts its own proxy (`proxy.py`) between the harness and the server. It
writes the shipped sampler into every request explicitly — zeros included, since an
omitted `min_p` is llama.cpp's 0.05, not chad's 0 — records every request with the
server's own token counts and timings, and turns each trial's log into its ATIF
trajectory (`proxy_atif.py`), so `trace.py` reads a foreign trial like chad's. A block
starts by checking, per endpoint, that the server applied the forced sampler, and by
emptying the server's slots, so no arm inherits another's prefix cache.

An entry is data: the harness's headless argv, the environment variables it reads, the
config files its home should hold. Every trial runs it cut off from the machine it runs
on:

- the workspace is under the system temp directory, not this checkout, because most
  harnesses read `AGENTS.md` from the directories above the one they start in;
- `HOME` and the XDG directories are a throwaway home the entry's config is written
  into, and the environment is rebuilt from an allowlist, so no key or token from your
  shell reaches the agent;
- the process runs under `sandbox-exec`, with file writes denied outside the workspace,
  that home and the temp directories;
- at the wall cap the harness's whole process group is killed.

chad in process is the one arm not isolated this way: MLX needs Metal, so it keeps the
real home, and chad sandboxes every bash command the model runs itself.

`harnesses.lock` pins the version of each harness a result is measured with, and
`run.py` refuses one that has drifted. `python benchmarks/polyglot/harnesses.py` lists
installed against locked; `harnesses.py lock <name>` pins what is installed. A CLI arm's
row has the same frame as chad's, with the harness's exit code where chad's own counters
are; its output is kept in `_runs/<label>/output/`.

## Publishing a run

Nothing a run produces is committed: `_runs/`, `_work/` and the rest of the underscore
directories stay on the machine, and a test fails if any of it is ever tracked. A run
that a write-up cites is published instead:

```sh
uv run python benchmarks/polyglot/publish.py --label baseline           # rows only
uv run python benchmarks/polyglot/publish.py --label baseline --with-trajectories --upload
```

`publish.py` rewrites every local path (the workspace becomes `.`, a CLI arm's throwaway
home `<home>`, the home directory `~`) and refuses the bundle if a home path, a path outside the workspace or a
credential-shaped string survives. It prints the sha256 of the rows and a row to paste
into [`RUNS.md`](RUNS.md). It uploads to the dataset repository only with `--upload`.
`fetch.py --label <label>` downloads a published run into `_runs/<label>/` and refuses it
unless its rows hash to what `RUNS.md` recorded, so someone else's baseline can be
compared against a local arm.

## Files

| file | job |
|---|---|
| `catalog.py` | pinned upstream, the task list, `manifest.json` loader |
| `workspace.py` | leak-free trial directories, `run-tests.sh`, `verify()` |
| `prepare.py` | one-time fetch of everything trials need from the network |
| `gold.py` | the gold gate; writes `manifest.json` |
| `run.py` | a block of trials: one arm, one model load, one row per trial |
| `harness/` | the contract every arm meets (`__init__.py`), chad in process (`chad_inprocess.py`), any CLI agent in isolation (`cli.py`) |
| `harnesses.py` | every CLI arm as data; `lock` pins the installed versions |
| `harnesses.lock` | the version, install command and entry-point sha256 each arm is measured with |
| `server.py` | the one llama-server of a llama phase, and the one-engine-at-a-time rule |
| `proxy.py` | the sampler forced on every request, and every request recorded with the server's counts |
| `proxy_atif.py` | a trial's request log as an ATIF trajectory; `crosscheck` against chad's own |
| `subsets/` | pre-registered task lists (`stats.py subset`), committed before any arm ran on them |
| `stats.py` | score, pool, paired compare |
| `trace.py` | one trial's ATIF trajectory as a step table: tokens, think, seconds, why each step ended — readable while the trial is still running |
| `publish.py` | a finished run as a path-free bundle, uploaded only on request |
| `fetch.py` | a published run back into `_runs/`, checked against its `RUNS.md` hash |
| `RUNS.md` | the ledger of published runs: what ran, where it is, the sha256 of its rows |

Each row records pass/fail, wall clock, steps, generated / thinking / prefilled tokens,
peak context, which levers fired, and the machine's thermal state. `meta.json` records
the chad version and commit, the model, the context limit and every `CHAD_*` variable.
