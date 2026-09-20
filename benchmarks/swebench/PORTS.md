# PORTS.md — every deliberate difference between the two arms

`bench/legacy` is the 2.2.0 release tree with the `c45c8dc` ("warp drive", 2026‑08‑18)
agent layer restored on top of it. This file is the ledger the plan asks for: every
commit that touched the agent path since the purge is classified once, in writing,
**before any scored trial**, and every difference the bridge finds is either justified
here or fixed.

The rule, from the plan:

> MODEL-DRIVING / BUGFIX → port (tool-call parsing, template/think handling, JSON
> repair, tab-indent fit, cached-id ledger — anything a user would call a bug in either
> design); DESIGN → never port (spill levers, prompts reworded for the lean surface,
> todo nudge).

Two categories the plan implies and this file names explicitly:

- **UNREACHABLE** — the behaviour cannot fire under this campaign's conditions. Declared
  here rather than discovered in the residuals.
- **KNOWN DIFFERENCE** — a real behavioural gap that is *not* ported, with the reason,
  the expected direction, and the number that will price it.

| | |
|---|---|
| release tree | `dd6d443` (chad 2.2.0) |
| legacy agent layer | `c45c8dc` |
| legacy tree | `bench/legacy` — see `PREREG.md` for the frozen sha |
| commits classified | 24 touching `src/chad/` in `0c9baf5..dd6d443` |

## The seam, and why the engine is untouched

`bench/legacy` = **HEAD's engine layer** (`engine.py`, `base_engine.py`,
`completion_engine.py`, `mlx_*.py`, `prism_pack.py`, `config.py`) + **`c45c8dc`'s agent
layer**. The plan asks for the diff across the `BaseEngine` seam measured in both
directions, taking the smaller. Measured:

| direction | cost | verdict |
|---|---|---|
| adapt the agent to 2.2.0's engine | 2 call sites in `agent.py` | rejected — it *silently removes* the sub-agent's cache quarantine, which prices the feature's implementation accident instead of its design |
| revert #60's engine hunks | ~156 lines in `engine.py` + 19 in `base_engine.py` | rejected — the engine must be identical by construction, or the arms differ by more than the harness |
| **move the quarantine to the agent side** | **one new module, 2 call sites** | **taken** |

The third option exists because the seam is one call wide. `Engine.push_cache` /
`pop_cache` swap six attributes and optionally spill to disk; **every one of those
attributes still exists at 2.2.0 under its 1.x name**, so the behaviour is a state swap
over a live class rather than a revert of a released refactor. It lives in
`chad/quarantine.py`, its coupling contract is written down as a `Protocol`
(`CachedEngine`), and a test asserts that contract against the real `Engine` — so if the
engine layer ever moves again, the failure is a red test and not a parent session that
comes back subtly wrong.

`config.py` is byte-identical between the two trees, so it counts as shared rather than
ported.

## PORTED — a bug is a bug in either design

| commit | what | why it is a bug, not a design choice |
|---|---|---|
| `39c07d1` (#36) | stray-`<think>` roundtrip | A `<think>` the model writes inside its own reasoning defeated both helpers at once: `close_unclosed_think` read it as "already a block" and skipped the close, and `split("<think>")[-1]` then dropped every reasoning token before it. The turn is stored unclosed, re-renders with an injected empty block, and diverges from the cache at the first token it generated. Reasoning the model emitted disappears from the transcript. The ported test pins the *property* — the next render must begin with the bytes already decoded — for each turn shape a 27B really emits. |
| `4256c38` (#40) | two-tier warm prefix | The system-prompt checkpoint was keyed on the whole rendered prefix, working directory included, so only a restart in the same project could hit it. **Every trial in a block starts in a directory that has never existed before**, so without the static head checkpoint the arm pays the full cold prefill fifty times a night. That is a cost of the checkpoint key, not of the design under test. Adds `prompt.static_system_prompt` and `Agent._static_head_ids`. |
| `cf196cd` (#58) | JSON repair edits structure only | The trailing-comma, Python-constant and bare-key regexes ran over string values, so a salvaged call ran `python -c 'assert x is null'` for what the model wrote, and a salvaged `write` put `x = true` in the file. The tool result looked normal. The arm has to run what the model wrote. |
| `ad25de9` (#66) | bounded zero-hit definition pointer | Cold, it parsed every code file in the repo synchronously between the tool call and its result — 17.5 s on an 11k-file repo, and **django is 11k files**. Now memoized per identifier, bounded at one second, and never re-walks. |
| `ad25de9` (#66) | repomap cache header + sweep | A stale cache file was unpickled in full before its version was checked, and nothing ever evicted one. A block opens fifty never-seen repos a night. Header line checked before the body; 30-day / 256 MB sweep; cache version 4. |
| `7a25eb5` (#68) | UTF-8 in every file tool | The tools used the locale encoding, and `edit` read with `errors="replace"` and then wrote the result back — one edit silently corrupting every non-UTF-8 byte in the rest of the file. **The sphinx half of this corpus ships test fixtures that are not UTF-8.** `edit` now refuses such a file with a clear message. |
| `7a25eb5` (#68) | bash head/tail band | The clip note counted against the body's budget, so a body just under the cap took the truncation path, `omitted` came out negative, and the head and tail slices overlapped — printing the middle twice under a notice saying it had been dropped. |
| `2370c88` (#64) | `_EXECUTES_RE` widening | The verify gate refused the everyday launch shapes of a real test run (`VAR=value` prefixes, `timeout [opts] DURATION`, `.venv/bin/python`, `/usr/bin/python3`, `coverage run`), so a model that **did** run its tests was sent back to run them again, and its `done` was hard-stopped once the nudges ran out. |
| `1b94d03` (#76) | `CHAD_SESSION_DIR` | The saved-session store was bound at import to `~/.chad/sessions`, so the suite wrote the developer's real home state. Three lines. |
| `f14ae3e` (#78) | one `_template_extra()` | `_render` built the `reasoning_effort` kwarg inline while `_stable_prefix_ids` rendered without it, so on a template that accepts it the checkpointed prefix was **not a prefix** of the turn it was meant to warm and the warm start missed with nothing to say so. It also reads the engine's carried default, which the shipped harness renders with and a 1.x agent would have ignored. |
| `c86ee8f` (#62) | `_under_plans` resolves symlinks | Used `abspath`, which does not resolve symlinks, so a `plans` entry that is a symlink carried an unconfirmed plan-mode write wherever it pointed. Unreachable in this campaign (see below); ported anyway because it is three lines and the tree gets run by hand. |

## NOT PORTED — design, which is the thing being measured

| commit | what | why it stays |
|---|---|---|
| `9c85e01` (#32) | the 2.0.0 lean purge | This **is** the design under test. |
| `2370c88` (#64) | strip the runner wrapper in the verify gate | The 1.x tree carries the same regex in `ambient.py` and its own comment declines to use it here: *"the verify gate's own regex is deliberately untouched (changing what disarms the unverified-edit flag is a different, riskier change)."* That is a 1.x design decision on the record, and the ledger keeps it. `uv run pytest` therefore does not clear the unverified-edit flag in this arm. |
| `2370c88` (#64) | revert un-lands edits; only executing commands verify | Already present in the 1.x tree, lever-gated (`revert_rearm_gate`, `verify_requires_execution`). #64 was re-landing in 2.x what 1.x already had. |
| `7a25eb5` (#68) | the `write_todos` prompt line | 2.x adds a line documenting the JSON shape. The 1.x JSON Schema already declares that shape, so the 1.x tree is not describing a format its tool rejects — this is added steering, not a correction. |
| `7a25eb5`, `53d3bf3` | `write_todos` and `edit` schema descriptions | 2.x adds steering text ("send the whole list every time", "set a step to `completed` ONLY after you have run something that shows it works", "copy the current text from what you just read"). Prompt design for the lean surface. The bridge reports both. |
| plan 142 | `trim_spill`, `result_spill` | 2.x-only levers, born with the lean surface's `spill.py`. The legacy arm has its own compaction spill (`compact_offload`). `spill.py` is deleted from the legacy tree: the 1.x compaction layer carries its own. |
| `53d3bf3` (#74) | writes outside the workspace always confirm | A 2.1.0 security policy. The runner's own sandbox confines writes to the trial workspace for **both** arms (`sandbox.py`), which is the same guarantee applied identically and outside either tree. Superseded rather than skipped. |
| `3c3a7ac` (#67) | lazy engine import in cli; idle TUI redraw | Startup and TUI only. `run.py` constructs `Engine` directly and no TUI runs. |
| `739a57c` (#46), `705a642` (#51) | `chad bench` fixes | The benchmark driver here is `run.py`, not `chad bench`. |
| `e2da687` (#69) | delete dead 1.x residue | The "residue" is the 1.x surface. Deleting it is the purge. |
| `b91ebbf`, `c16c9a6`, `0ed7a02`, `59e4838`, `09dcda2` | gate, mypy, docs | No behaviour. The legacy tree is held to the release tree's gate anyway: ruff, mypy and anti-slop all pass on it. |

## UNREACHABLE — declared before scoring

| what | why it cannot fire |
|---|---|
| `chad serve` | Needs the server module the 2.2.0 engine layer no longer ships, and the plan forbids a server outright — the premise is one Mac running the model in-process. Removed from `cli._SUBCOMMANDS` and pinned gone by a test. |
| plan mode, and `plan_review` | Every trial runs `mode="yolo"`. |
| `yolo_seatbelt`, `seatbelt_protect_git` | Seatbelt profiles do not nest; the runner owns confinement for both arms. `run.py` sets `CHAD_NO_SEATBELT=1`. |
| `profile_prompt` | Measured: `profiles.resolve()` returns the `qwen38` profile for the shipped pack and that profile's block is empty by design. Contributes 0 characters. |
| `catalog_clip`, `activate_skill` | Skill discovery is off for both arms — see below. |
| `backend_retry` | Retries a transient 5xx from a remote backend; the engine is in-process. |
| `ts_edit_revert` | Extends the Python revert to other parseable languages; both repos are pure Python, so `syntax_revert` covers every file the arm can edit. |

`legacy-manifest.tsv` carries the same predictions per lever, in a `fires_here` column,
frozen by sha before the first scored trial.

## KNOWN DIFFERENCE — not ported, measured instead

**A call batched with `done` is dropped** (`2370c88`, #64). The 2.x loop routes it
through the ordinary per-call path; the 1.x loop returns before the execute loop, so an
`edit` (or the habitual closing `write_todos`) the model emitted alongside `done` never
runs, and the model is then told it has not done anything.

Not ported because the fix is a restructure of ~120 lines of the turn loop, which is
likelier to introduce a difference between the arms than to remove one. Counted instead:
`Agent.dropped_sibling_calls` increments per dropped call, `run.py` writes it into every
`trials.jsonl` row (0 on the lean arm), and Phase 4 prices it. **Direction is known in
advance: it can only ever cost the legacy arm.** If the count is zero the difference did
not bite; if it is not, the analysis reports resolve rate with and without the affected
trials.

## Five findings from building this, all of which change the runner

**1. The operator's personal skills were riding into every prompt.** Skill discovery
walks `~/.claude/skills` as well as the workspace. On the machine this campaign runs on
that is 70 skills, and the legacy arm's system prompt goes **7,655 → 33,450 characters** —
three quarters of it a catalog that exists on no other machine and has nothing to do
with the harness under test. It also adds a tool (`activate_skill`), so the arm's surface
would differ per operator. The 1.x tree's own code comments this exact hazard: *"a
measured confound: ~50 user skills were injected into every prompt … shifting a small
model's greedy trajectory."*

`run.py` now sets `CHAD_NO_SKILLS=1` for both arms, before chad is imported. The release
arm is unaffected either way (the purge took skills out of its prompt path); the legacy
arm's surface is therefore **21 tools, not 22** — `activate_skill` exists only when a
skills directory does. Every count in the kit says 21.

**2. The unattended-run governors had no deadline.** `wrapup_window`, `hard_wrapup`,
`turn_think_budget` and `late_continue_replenish` take their budget from
`CHAD_TURN_BUDGET_S`, which nothing set — so four levers in the legacy arm's own
manifest were configured on and structurally unable to fire, while the runner's
`should_stop` cut trials off mid-generation instead. `run.py` now sets it to the trial
wall cap, for both arms (both trees read it; only the legacy lever set acts on it).

**3. The runner's own scratch was riding into the prediction.** The trial's `TMPDIR`
sits inside the workspace, because the sandbox profile makes the workspace the one place
a trial can certainly write — which puts it one `git add -A` away from the patch. On the
first live legacy trial, node (spawned by the language server) left **81 compile-cache
files** there and `workspace.diff` handed back a **207 KB prediction with no source
change in it at all**. It is also arm-asymmetric: only the legacy arm runs a language
server, so only the legacy arm spawns node.

`workspace.SCRATCH` now names the directory next to the exclusion that keeps it out:
`materialize` writes it to `.git/info/exclude` before the seed commit, so it is invisible
to `diff()` and to the `git status` the model reads, and `diff()` asserts the exclusion
held rather than trusting it. A kit test reproduces the leak and pins the fix.

**4. H2's first-token number was `None` on every trial.** The runner recorded it off an
emit of kind `stream`, and a block calls `run_turn(stream=False)` — which builds no
stream view, so nothing of that kind is ever emitted. Both arms would have reported null
for the whole campaign, for the one cost number a user actually feels. It now keys off
`gen`, the throttled decode counter both trees emit identically every 16 tokens, and the
field is named `first_gen_s` for what it measures: at this model's decode rate those 16
tokens are about a second against a cold prefill of tens.

**5. The on-disk warm start was off, so every trial cold-prefilled its whole prompt.**
`Engine.cache_dir` defaults to `None`, which disables KV checkpointing outright; the
shipped CLI sets it and the runner did not. Measured on a live legacy trial: **194 s to
the first decoded token**, nearly all of it prefill of a prompt whose static head never
changes. It also made the two-tier warm-prefix port above inert — the port exists
precisely because every trial stands in a directory that has never existed.

The runner now builds the engine the way `cli.main` does, with a **kit-owned** store
(`benchmarks/swebench/_kv`, 8 GB LRU, the CLI's own default) rather than
`~/.cache/chad/kv`, which on this machine already holds 7.8 GB from unrelated work — a
block must not start warm or cold depending on what the operator did last week.
`meta.json` records the store and its size at block start, because block 1 of a campaign
pays a head prefill that block 2 does not and the analysis should see that rather than
average over it. `--no-kv-cache` keeps the old behaviour available as an A/B arm.

## The bridge

`bridge.py` renders the first request of a fixed task in both trees — `bench/legacy`
under `CHAD_LEAN=1` and the release tree — through each tree's own tokenizer, chat
template, prompt builder, schema builder and sampler preset, in separate subprocesses so
neither `chad` package can shadow the other.

**Stage 1, as shipped.** Not identical, and the differences are exactly the
prompt-and-schema text listed under NOT PORTED: `result_spill`/`trim_spill` in the lever
set, the `edit` and `write_todos` descriptions, and the `write_todos` prompt line. Token
ids diverge at id 74 of 2,265 / 2,442 — inside the schema block, which is what the
template renders first.

**Stage 2, controlled.** Hold the system prompt and the tool schemas equal and re-render
in each tree: **2,442 token ids, identical**, samplers identical. The template, the
tokenizer, the id coercion and the sampler preset are therefore shared, and the stage-1
difference is design and nothing else.

That is the claim the plan needed the bridge to support, and it is stronger than a pass
on the as-shipped render would have been: a byte-identical stage 1 would only have said
the two prompts happened to agree, whereas stage 2 isolates the machinery from the design
and shows the machinery is the same code.

Re-run it after any port:

```
uv run python benchmarks/swebench/bridge.py --legacy ../chad-legacy [--diff]
```

## The tool surface, exercised

`tool_probe.py` calls every tool in the legacy arm's dispatch table once, against a real
`django/django` checkout at a real base commit, inside the same sandbox profile and the
same workbench environment a trial gets, and asserts on the *content* of each result —
a `find_refs` that returns "no references" is not a working tool.

**19/19 dispatchable tools work** (`done` and `task` are dispatched by the agent loop
rather than the table, so they are checked for schema presence instead). The
language-server tier is live: pyright resolves from the uv cache with no network,
indexes django, and answers `definition`, `find_refs`, `hover` and `rename_symbol`
precisely — 32 references for `Paginator`, resolved in 0.5 s.

Two things this found that a README would not have:

- The trial sandbox's `read_roots` are load-bearing. Without them a uv-managed
  interpreter cannot read its own venv and dies with `Failed to import the site module`.
- `uv run` prepends its project's venv to `PATH`, so a `PATH` set in the *parent's*
  environment ends up second and `python` resolves to the harness interpreter rather
  than the workbench's. Both the probe and `run.py` set `PATH` from inside the process.

```
uv run python benchmarks/swebench/tool_probe.py --tree ../chad-legacy
```
