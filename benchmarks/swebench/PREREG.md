# Pre-registration — lean vs legacy harness design on SWE-bench Verified Mini

**Status: DRAFT.** Everything marked `TBD` is fixed by the LEAN pilot, before the first
LEGACY trial is scored, and this file is committed in its final form at that point. The
write-up links that commit. Nothing below is revised afterwards; if something has to
change, the change is added as a dated amendment at the bottom with its reason, and both
versions stay in the history.

## The question

With the model, the engine, the sampler, the tasks and the machine held fixed, does the
full 1.x harness surface (22 tools, an LSP/symbol layer, a sub-agent, 69 levers) resolve
more SWE-bench instances than the 2.x loop (5 tools, 10 levers on by default)?

**Thesis under test.** Self-scaffolding works when task, scaffold and policy are trained
together under one reward. Absent training, the remaining move is to lean the harness
into what the model already does. 1.x pushed against the model; 2.x stopped pushing.

## Arms

| arm | tree | surface | levers |
|---|---|---|---|
| **LEAN** | release tag, frozen (sha `TBD`) | 5 tools | the 10 that are on by default |
| **LEGACY-FULL** | `bench/legacy` (sha `TBD`) | 22 tools | `legacy-manifest.tsv`, frozen by sha |
| LEGACY-TOOLS (follow-up) | `bench/legacy` | 22 tools | all off — the 1.x shipped default |

Primary comparison is LEAN vs LEGACY-FULL at k=`TBD` (target 5). LEGACY-TOOLS runs
afterwards and only on the instances where those two disagree or either is unstable: it
explains a difference, it is not needed to find one.

**Steelman rule.** LEGACY-FULL gets its best honest configuration. `CHAD_ENABLE=all` is
not a configuration; the manifest names every lever ON or OFF with a reason, and the
runner refuses the arm without it. Any feature that cannot fire on a host checkout is
listed as unreachable *before* scoring, never discovered afterwards.

## Held fixed across arms

Weights (the shipped ternary Qwen3.8-27B pack with its DFlash2 drafter), engine code,
sampler, task prompt, wall cap, step cap, workbench environment, sandbox profile,
workspace construction, and task order within a replicate. The runner owns all of them;
neither tree is asked to configure itself. `meta.json` records the resolved values for
every block, including the actual sampler read back off the engine.

**Sampler.** Each tree's own shipped preset helper is applied (`apply_sampler_preset`
then `apply_sampler_env`), so the sampler is what a user gets rather than one invented
for the benchmark. On the current release that is thinking mode at temp 1.0, top_p 0.95,
top_k 20, min_p 0.0. It is recorded per block and is identical across arms by
construction.

**Prompt.** `problem_statement` verbatim, followed by one fixed paragraph
(`run.py: PROMPT_SUFFIX`) that says what a solution is and that tests must not be edited.
It names no file, no test command and no tool. `hints_text` is never shown — it is
dropped at fetch time and never written to disk.

## Hypotheses

**H1 (primary, equivalence).** LEGACY-FULL does not resolve more instances than LEAN by
more than δ.

- Estimator: the mean over the 50 instances of (p̂_legacy − p̂_lean), where p̂ is an
  instance's resolve rate over the k replicates.
- Interval: instance-clustered bootstrap.
- Test: TOST against ±δ, reported alongside the one-sided read.
- **δ = TBD, expected ≈ 10 pp**, fixed after the LEAN pilot by simulating from its
  per-instance rates. 50 instances at k=5 on one laptop cannot honestly claim tighter,
  and the write-up will not.
- "No significant difference" is never the claim. Either the interval excludes δ or it
  does not, and either way the interval is printed.

**H2 (cost on a laptop).** Per trial: prefill tokens, schema + system prompt tokens,
round trips, wall clock, capped rate, seconds to first token. Plus the figure the premise
deserves: **resolve rate as a function of wall budget**, per arm.

**H3 (reachability).** Tool-call share under the 22-tool menu — is bash still ~80% of
calls when `find_refs` is on offer? — and per-lever fire counts.

**H4 (localization).** Did the submitted patch touch the gold files? File-level hit rate
per arm, computed by joining `changed_files` against the instance rows at analysis time.
This is the legacy navigation layer's best case.

**H5 (attribution).** Fired-conditioned paired flips for every tool and lever that fires
often enough, classified causal / incidental / fired-null / unexercised.

**Null calibration.** Within-arm label permutation, and LEAN split-halves, reported
beside every delta.

## Grading

Official `swebench.harness.run_evaluation` (pinned release in `grade.py: HARNESS`)
against `princeton-nlp/SWE-bench_Verified`, restricted to the 50 mini instance ids.
Empty or malformed patches score as unresolved. Nothing in this kit decides whether an
instance is resolved.

**Route: the published x86_64 images under emulation**, recorded in every report.
Building the images locally for arm64 was the plan's first choice and is measured not to
work: an image built today installs the project with no date pin, so sphinx gets a 2026
docutils and its suite dies during collection. See `grade.py`'s module docstring.

**Gold gate: PASSED 50/50** on that route, 2026-09-19, before any agent trial
(`_runs/_gate/gold.gold-x86_64-20260919.json`). Re-run it if the harness pin, the image
route, or the machine changes.

## Disclosed departures from a server-scale run

- **Wall cap per trial** (`TBD`, fixed after the pilot). SWE-bench has no official
  timeout, so this one is ours. A capped trial submits whatever diff exists.
- **Step cap** per trial, same for both arms, recorded per block.
- **Native workbench, official grade.** The agent works in a host environment that
  differs from the container in named ways (`envspec.py`); the grade always comes from
  the container. Both arms see the same workbench.
- **Workbench validation is not a filter.** All 50 stay in the scored set. A sensitivity
  analysis restricted to the instances that validated is reported beside the headline.
- **Contamination.** Not fixable on this benchmark; it shrinks the discriminating band
  and is stated in the write-up rather than worked around.

## Stop conditions

- Gold gate below 50/50 on both image routes → the grader here is not the public grader.
  Stop.
- Workbench validation below ~40/50 → fix environments before spending nights.
- Leak canary reaches the network, `$HOME`, or a planted fix → no trials until it cannot.
- The static bridge between the trees cannot be made clean → run LEAN from `bench/legacy`
  with the lean flag, amend this file, and say so in the write-up.
- Pilot LEAN resolve rate outside ~15–85% → the set cannot discriminate for this model.
  Stop and rethink before spending a month of nights.
- Any panic, freeze, or thermal throttling pattern → halt the campaign, not just the
  block.

## Commitment

Publish whichever way it lands. If LEGACY-FULL wins by more than δ, the write-up is "we
purged too much" and the features come back.

## Amendments

_None yet._
