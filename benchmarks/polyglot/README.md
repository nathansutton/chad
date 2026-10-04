# Polyglot: chad's coding eval, reproducible on one Mac

215 hard Exercism exercises in C++, Go, Java, JavaScript, Python and Rust. The agent gets
the problem statement and a stub, works in a real directory with a real toolchain, and
passes only if the exercise's full test suite exits 0. Everything runs on the laptop the
model runs on: no Docker, no server, no grader in the cloud.

The exercises are [Aider's polyglot benchmark](https://github.com/Aider-AI/polyglot-benchmark),
pinned to one upstream commit and fetched rather than vendored. The numbers are **not
comparable to Aider's leaderboard**: that protocol is two attempts at a whole-file edit,
and this one is an agent loop that may run the tests as often as its budget allows.

An eval is only useful if it can tell two versions of the agent apart. chad's hand-built
task tiers cannot any more, since the shipped model passes all of them. These exercises
were selected upstream because strong models fail them, and they cost minutes each, so a
laptop can afford the repetitions that statistics need.

## Run it

```sh
uv run python benchmarks/polyglot/prepare.py     # once: exercises, jars, jest, crates
uv run python benchmarks/polyglot/gold.py        # once: prove the toolchains, no model
uv run python benchmarks/polyglot/run.py --label baseline --reps 3
python benchmarks/polyglot/stats.py score benchmarks/polyglot/_runs/baseline/trials.jsonl
python benchmarks/polyglot/trace.py benchmarks/polyglot/_runs/baseline/trajectories/go/book-store.rep1.json   # one trial, step by step
```

Toolchains: `python3`, `go`, `cargo`, `node`+`npm`, a JDK (17+), `cmake` and a C++
compiler. `prepare.py` names any that are missing; their tasks are skipped, not failed. A
block loads the weights once and is resumable after a crash or a closed lid. Budget four
to five minutes a task on a 24 GB M4 Pro: about 16 hours for one pass over all 215.

## What keeps a score honest

- **The gold gate.** Every exercise's untouched stub must fail and upstream's reference
  solution must pass. 215 of 225 clear both and are listed in `manifest.json`; the other
  10 are excluded with the reason.
- **No answer in the workspace.** `.meta/`, `.approaches/` and the hints file never reach
  the trial directory or the prompt.
- **Tests belong to the harness.** Test files, build files and `run-tests.sh` are restored
  from upstream before grading.
- **The whole suite counts.** Upstream ships most tests disabled; all are switched on.
- **Exit status decides.** No output parsing.
- **Offline trials.** Dependencies are fetched up front and pinned; nothing a trial runs
  reaches the network.
- **It is the shipped agent.** Shipped model, sampler and context limit. The one cap the
  benchmark adds is wall clock per trial (`--wall-cap`, 1200 s), recorded as `capped`.

## Comparing two versions

Pass rates of two runs differ mostly because tasks differ in hardness. `stats.py compare`
pairs by task and runs an exact sign test on the tasks whose pass rate moved, so
significance depends on how many tasks flipped, not how many ran; it needs at least six
one-directional flips to reach p < 0.05. To make flips cheap, pin a **pool** from a
baseline with two or more reps, the tasks it passes sometimes but not always:

```sh
cd benchmarks/polyglot
python stats.py pool _runs/baseline/trials.jsonl > _runs/baseline/pool.txt
CHAD_DISABLE=env_manifest uv run python run.py \
    --label no-manifest --tasks-file _runs/baseline/pool.txt --reps 3
python stats.py compare _runs/baseline/trials.jsonl _runs/no-manifest/trials.jsonl
```

The pool authorizes a change; the full set only vetoes one.

## Other harnesses

`--harness` runs the same tasks, prompt, wall cap and verification under another coding
agent's own command line: an entry of `harnesses.py` (pi, opencode, dsh, goose,
mini-swe-agent, crush, cline, codex), or `chad-llama`, chad's own CLI on the same server,
which is the arm every foreign one is paired against. They all talk to one llama-server
serving **the file chad loads**, byte for byte, so chad in process against chad on
llama-server is an engine comparison and a foreign harness against chad on llama-server
is a harness comparison:

```sh
uv run python benchmarks/polyglot/server.py        # chad's own GGUF in llama.cpp; Ctrl-C stops it
uv run python benchmarks/polyglot/run.py --label h3-pi --harness pi \
    --tasks-file benchmarks/polyglot/subsets/harness-3.txt --reps 3
python benchmarks/polyglot/stats.py scorecard --ref _runs/h3-chad-llama _runs/h3-*
```

The scorecard reads each arm the way the harness article did: the prompt tax of turn 1,
the tokens the prefix cache could not serve on later turns and their wait, cache reuse,
side requests, and pass rate paired with the reference arm by task.

Each CLI block puts a proxy (`proxy.py`) between the harness and the server. It writes the
shipped sampler into every request explicitly (zeros included, since an omitted `min_p`
is llama.cpp's 0.05, not chad's 0), records every request with the server's own counts,
and turns each trial's log into an ATIF trajectory so `trace.py` reads a foreign trial
like chad's. Every trial runs cut off from the machine: a workspace under the system temp
directory, a throwaway `HOME` with the entry's config written into it, an environment
rebuilt from an allowlist, `sandbox-exec` denying writes elsewhere, and the whole process
group killed at the wall cap. chad in process is the one arm not isolated this way, since
MLX needs Metal; it sandboxes every bash command the model runs itself.

`harnesses.lock` pins the version each harness is measured with and `run.py` refuses one
that has drifted; `harnesses.py lock <name>` pins what is installed. One engine at a time
is enforced: an in-process block refuses while a llama-server is up.

## Publishing a run

Nothing a run produces is committed: `_runs/`, `_work/` and the other underscore
directories stay on the machine, and a test fails if any of it is ever tracked. A run a
write-up cites is published instead:

```sh
uv run python benchmarks/polyglot/publish.py --label baseline                              # rows only
uv run python benchmarks/polyglot/publish.py --label baseline --with-trajectories --upload
```

`publish.py` rewrites every local path and refuses the bundle if a home path or a
credential-shaped string survives, then prints the sha256 of the rows and a row to paste
into [`RUNS.md`](RUNS.md). Published runs live in the public dataset
[`nathansutton/chad-polyglot-runs`](https://huggingface.co/datasets/nathansutton/chad-polyglot-runs);
`fetch.py --label <label>` downloads one into `_runs/<label>/` and refuses it unless its
rows hash to what `RUNS.md` recorded.

## Files

| file | job |
|---|---|
| `catalog.py` | pinned upstream, the task list, `manifest.json` loader |
| `workspace.py` | leak-free trial directories, `run-tests.sh`, `verify()` |
| `prepare.py` | one-time fetch of everything trials need |
| `gold.py` | the gold gate; writes `manifest.json` |
| `run.py` | a block of trials: one arm, one model load, one row per trial |
| `harness/` | the contract every arm meets, chad in process, any CLI agent in isolation |
| `harnesses.py` | every CLI arm as data; `lock` pins the installed versions |
| `harnesses.lock` | the version, install command and entry-point sha256 of each arm |
| `server.py` | the one llama-server of a llama phase |
| `proxy.py` | the sampler forced on every request, and every request recorded |
| `proxy_atif.py` | a trial's request log as an ATIF trajectory |
| `subsets/` | pre-registered task lists, committed before any arm ran on them |
| `stats.py` | score, pool, paired compare, scorecard, subset |
| `trace.py` | one trial's trajectory as a step table, readable while it runs |
| `publish.py` | a finished run as a path-free bundle |
| `fetch.py` | a published run back into `_runs/`, checked against its `RUNS.md` hash |
| `RUNS.md` | the ledger of published runs |

Each row records pass/fail, wall clock, steps, generated / thinking / prefilled tokens,
peak context, which levers fired, and the machine's thermal state. `meta.json` records the
chad version and commit, the model, the context limit and every `CHAD_*` variable.
