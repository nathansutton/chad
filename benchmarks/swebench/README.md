# SWE-bench Verified Mini, on one 24 GB MacBook Pro

This kit answers one question, and it is a question about harness *design*, not about a
model: when the model, the engine, the sampler and the tasks are all held fixed, does a
large agent surface — many tools, a symbol layer, a sub-agent, dozens of behavioural
levers — get more done than a bare loop with five tools?

It is chad against chad. Same weights, same engine code, same machine, same 50 tasks.
The only thing that differs between the arms is the harness.

Everything here runs on one Apple Silicon laptop, in-process, on the shipped MLX engine.
No server, no second machine.

## Why this benchmark fits a laptop

SWE-bench separates **solving** from **grading**, and that separation is what makes a
24 GB machine enough.

The agent solves on the host: a real checkout of django or sphinx, a real Python
environment, real tools, the model resident on the GPU. It emits a patch. Later, with no
model loaded, the official `swebench` harness grades those patches inside Docker exactly
as it would grade anyone else's submission. The weights and a grading VM never share the
memory — a rule the runner enforces rather than remembers: `run.py` refuses to start
while Docker is up, and `grade.py` refuses to start while a solving block is alive.

The benchmark is [`MariusHobbhahn/swe-bench-verified-mini`][mini], a published
50-instance subset of SWE-bench Verified, pinned here by revision and by the sha256 of
its parquet. 25 django/django (all version 3.1, so one environment) and 25
sphinx-doc/sphinx across 12 versions. Gold patches touch one file in 41 of the 50.

[mini]: https://huggingface.co/datasets/MariusHobbhahn/swe-bench-verified-mini

**Contamination, said plainly.** These issues are from 2017–2023 and both repositories
are famous. The model has very likely seen them, and probably some of the fixes. A paired
chad-vs-chad delta cancels most of that — a memorized task lands the same way for both
arms — but it costs discrimination: a task the model simply knows carries no information
about the harness, so the effective sample is smaller than 50.

## Running it

```sh
uv run python benchmarks/swebench/prepare.py all                   # no model, no Docker
uv run python benchmarks/swebench/grade.py gate                    # Docker; the gold gate
uv run --project . python benchmarks/swebench/run.py --arm lean --rep 1
uv run python benchmarks/swebench/grade.py run benchmarks/swebench/_runs/lean-rep1-*
```

`prepare.py` has six steps, each runnable alone and each safe to re-run: `dataset`,
`specs`, `clone`, `envs`, `canary`, `validate`.

A block runs under **its own tree's interpreter**, and `run.py` refuses to start
otherwise. The two trees do not have the same dependencies — the legacy agent layer
needs rustworkx for repomap's ranking, which the release tree dropped along with
`repo_map` — so the wrong interpreter either fails deep in an import or, worse, succeeds
and measures one tree's code against the other tree's packages.

```sh
# the legacy arm, its tree, its interpreter, its pre-registered lever manifest
uv run --project ../chad-legacy python benchmarks/swebench/run.py \
    --arm legacy-full --rep 1 --tree ../chad-legacy
```

Three checks belong to the legacy arm and are re-run after any port:

```sh
uv run python benchmarks/swebench/bridge.py --legacy ../chad-legacy      # the static bridge
uv run python benchmarks/swebench/tool_probe.py --tree ../chad-legacy    # every tool, on django
uv run --project ../chad-legacy python benchmarks/swebench/make_manifest.py \
    --tree ../chad-legacy --out benchmarks/swebench/legacy-manifest.tsv
```

## The five things that make the two arms comparable

Each is owned by the runner, not by either harness tree, so neither tree needs to know
the measurement exists.

**1. Workspaces with no future.** A trial does not get a clone. `git archive` exports the
tree at the base commit, and the destination gets a fresh `git init` and exactly one
commit — no remote, no later history, nothing in the object store that postdates the
task. In an ordinary clone of django, `git log --all` contains the fix. `git` still works
normally inside the workspace, because taking it away would be a harness difference
rather than a fixed environment. The one-commit property is asserted on every trial, not
once in a test.

**2. A sandbox the runner owns.** Stricter than chad's own: no network (loopback stays
open so a test can bind a port), writes confined to the trial's workspace and temp, and
reads of `$HOME` denied — so a newer django or sphinx anywhere on this disk, including
this repository, is unreadable from inside a trial.

The confinement is applied to the shell child, not to the block. Wrapping the whole block
would be simpler and does not work: the model runs in-process on Metal, and a process
inside a Seatbelt profile does not get GPU access — the load hangs. chad's own seatbelt
module carries the same rule for the same reason.

The injection point is `PATH`. Both trees resolve their shell once, at import, with
`which("bash")`; a directory placed first on `PATH` containing a `bash` that re-execs
under `sandbox-exec` therefore confines both arms identically, through code neither tree
contains. Seatbelt confinement is inherited and cannot be dropped, so the first shell is
the only one that needs wrapping. The shim fails closed — with no profile named it
refuses to run rather than falling through to an unconfined shell.

`prepare.py canary` proves the boundary in both directions before any trial: five things
that must be denied (DNS, a raw IP, `$HOME`, a planted "fix" file by exact path, a write
outside the workspace) and four that must be allowed (reading and writing the workspace,
writing temp, binding a local port). A profile that denies everything is not a boundary,
it is a broken shell, and it would fail every trial in a way that looks like the model
being bad at its job.

**3. Native workbench environments.** One per (repo, version) — 13 in all — built from
SWE-bench's own recipes, which `prepare.py specs` dumps verbatim to
`envs/upstream_specs.json` from a pinned `swebench` release. Every place the native
environment differs from the container is named in `envspec.py`'s module docstring, with
its reason, and each lockfile in `envs/` records the interpreter and the resolution date.
The differences that matter:

- django 3.1 runs on Python 3.8 (upstream asks for 3.6, which has no arm64 build).
- The package under test is never installed; `PYTHONPATH` points at the trial's own
  workspace, which is what an editable install would give it in the container.
- **Dependencies resolve as of the environment-setup commit's date.** Without this,
  nothing in the sphinx half runs: sphinx 3.x asks for `docutils>=0.12` with no upper
  bound, so resolving today puts docutils 0.23 into a 2020 project and the suite dies
  during collection. The published images do not have this problem because their
  resolution was frozen the day they were built; ours is frozen too, at a date derived
  from the data rather than chosen.

**4. A validation gate on the workbench itself.** Per instance, at the base commit with
the official test patch applied: FAIL_TO_PASS must fail, a sample of PASS_TO_PASS must
pass, and with the gold patch applied FAIL_TO_PASS must pass — all of it run under the
same sandbox a trial gets, so an environment that only works unconfined is caught here
rather than in the campaign.

It is a diagnostic, not a filter. All 50 stay in the scored set whatever it says, because
the official grade comes from the container; dropping tasks because our workbench is
imperfect would quietly redefine the benchmark. The analysis reports a sensitivity check
restricted to the instances that validated.

**5. One model load per block.** A block is one arm, one replicate, 50 trials, weights
resident throughout — about ten loads for a whole campaign rather than five hundred. Arms
interleave by block (L F / F L / L F …) so heat, OS updates and the calendar fall on both
arms equally, and a campaign cut short is still balanced.

## What is recorded

`predictions.jsonl` is exactly the official three-field shape and nothing else, so it
goes to the grader untouched. Everything the measurement actually cares about goes to
`trials.jsonl` beside it: prefill tokens, peak context, round trips, wall clock, whether
the wall cap bit, seconds to first token, which tools were called and how often, which
levers fired, and the files the submitted patch touched.

Nothing in the solving path can read the gold patch. `prepare.py dataset` writes two
files — `instances.jsonl` with everything, and `tasks.jsonl` with the problem statement
and nothing else — and `run.py` imports only the loader for the second. File-level
localization is scored later by joining the recorded `changed_files` against the instance
rows. `hints_text` is dropped at fetch time; the field never lands on this disk.

## Phase 0 status, measured on this machine

- **Dataset** pinned and split: 50 instances, 13 environments, digest verified.
- **Sandbox canary**: 9/9 probes behave as declared.
- **Environments**: 13/13 build.
- **Workbench validation**: **40/50** instances reproduce the benchmark's pass/fail
  structure natively. The rows are in `_data/workbench_validation.jsonl`.
- **Gold gate: PASSED, 50/50** on the published x86_64 images under emulation
  (`_runs/_gate/`). The official harness reproduces every gold patch here, which is what
  licenses any later number. Nothing agentic is scored until this holds.

What the gate cost, since the campaign shares a laptop disk with 12 GB of weights:
**19.1 GB** of image store for all 50 instances — far less than 50 × 2.1 GB, because the
per-repo base and environment layers are shared — and **5m15s** to evaluate all 50 at
four workers. Emulation is much cheaper here than feared; pulling the images, not running
them, is the slow part, and it happens once.

The 10 that do not are concentrated in older sphinx (3.1–4.1) and one django instance:

- `django__django-12209` — every one of its FAIL_TO_PASS entries is a *docstring*, not a
  test name. django prints each test's docstring at `--verbosity 2` and upstream's log
  parser captured some of those lines as entries in their own right. The official grader
  never notices, because it matches these strings against the log of a whole-suite run;
  running tests by name does. 41 such entries are dropped across the 50 instances, and
  the count is recorded per instance rather than hidden. This instance has nothing left.
- The nine sphinx instances fail on tests the environment cannot support rather than on
  the change under test — one identified cause is `_testcapi`, which uv's managed CPython
  strips from the standard library and `tests/test_util_inspect.py` imports.

None of this changes the scored set, and none of it favours an arm: both arms get the
same workbench, and the grade always comes from the container.

## The grading route: the fallback turned out to be the only route

The plan's first choice was to build the grading images here for arm64 and keep the
published x86_64 images under emulation as a fallback. Measured, the first choice does
not work, for the same reason the native workbench needed a date pin.

A locally built image is built *today*, and SWE-bench's recipe installs the project with
`pip install -e .[test]` and no date pin — so sphinx's unbounded `docutils>=0.12`
resolves to a 2026 docutils and the suite dies on `No module named
'docutils.utils.roman'`. `envspec.py` fixes that for the workbench with
`--exclude-newer`; it cannot be fixed for the grader, because patching upstream's image
build is how a grader stops being the public grader.

Probed on two gold patches: **locally built arm64 1/2, published x86_64 2/2**, the
failure being sphinx. The published route then passed the full gate 50/50. So the route
is `--route x86_64` (the default) — images published back when those dependency graphs
still resolved, run under emulation. `--route arm64` is kept so the finding can be
reproduced rather than taken on trust.

Two practical notes. The published images must be pulled with an explicit
`--platform linux/amd64`; without it Docker reports "no matching manifest for
linux/arm64/v8" and the harness then fails each instance with a 404 *during* the run, so
`grade.py` pulls them up front instead. And published **arm64** images do exist on
Docker Hub and pull natively — but neither swebench 4.x nor 5.x can address them, as
both hardcode the image architecture and expose no flag for it. Renaming an arm64 image
to the x86_64 name the harness expects would work; it is not done.

## Phase 1 status: the legacy tree

`bench/legacy` is the 2.2.0 release tree carrying the `c45c8dc` agent layer — 21 tools,
65 levers, the LSP/symbol tier, `repo_map`, profiles and the sub-agent — on a **byte-
identical engine layer**. `PORTS.md` is the ledger: what was ported and why, what was
not, what cannot fire here, and the one known difference that is counted rather than
fixed.

| Phase 1 item | Result |
|---|---|
| `bench/legacy` builds and passes the release gate | ruff, mypy, anti-slop, 1282 tests |
| `BaseEngine` seam | one call wide; the quarantine moved to the agent side, engine untouched |
| `PORTS.md` | 24 agent-path commits classified; 11 ported, 10 design, 7 unreachable, 1 counted |
| `legacy-manifest.tsv` | 65 levers: 63 ON, 2 OFF (both seatbelt conflicts), 7 predicted unable to fire |
| static bridge | **clean on machinery** — 2,442 token ids identical under a controlled render |
| every tool exercised on django | **19/19** dispatchable tools, under the trial sandbox |
| `run.py` drives both trees | yes, and refuses the wrong interpreter |

Five findings from Phase 1 changed the runner, and every one of them would have been
invisible in the results:

- **Skill discovery was importing the operator's own `~/.claude/skills`** into every
  prompt — 70 of them on this machine, taking the legacy arm's system prompt from 7,655
  to 33,450 characters and adding a tool. `CHAD_NO_SKILLS=1` for both arms now. The
  legacy surface is therefore **21 tools**, not 22: `activate_skill` exists only when a
  skills directory does.
- **The unattended-run governors had no deadline.** Four levers the manifest turns on
  read `CHAD_TURN_BUDGET_S`, which nothing set, so they were configured on and
  structurally unable to fire. `run.py` sets it to the trial wall cap for both arms.
- **The runner's own scratch was riding into the prediction.** The first live legacy
  trial submitted a 207 KB patch of 81 node compile-cache files and no source change:
  the trial `TMPDIR` lives inside the workspace, one `git add -A` from the diff. The
  workspace now excludes it before the seed commit, and `diff()` asserts that it did.
- **H2's first-token number was `None` on every trial.** It was recorded off a `stream`
  emit, and a block runs `run_turn(stream=False)`, which emits none. It now keys off the
  decode counter both trees emit identically, as `first_gen_s`.
- **The on-disk warm start was off.** `Engine.cache_dir` defaults to `None`; the shipped
  CLI sets it, the runner did not. A live legacy trial took **194 s to its first token**,
  nearly all prefill of a static prompt head. The runner now uses a kit-owned KV store,
  and records its size at block start.

## Still to come

Phase 2, the pilot: LEAN k=1 on all 50, which fixes the wall cap, δ and k, and commits
`PREREG.md` before the first scored legacy trial.
