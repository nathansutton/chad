# Changelog

Notable, user-visible changes.

## Unreleased

- **24 GB Macs now get the 35B.** The 35B's floor was 32 GB because its working set
  SIGKILLed a 24 GB machine mid-turn. The fused attention kernel and the
  8-bit-from-the-start KV cache it enables cut the per-token cache cost enough to give the
  headroom back, and the compaction trigger sizes itself from the live Metal budget rather
  than a fixed constant, so a tight box narrows its window instead of dying. 16/18 GB
  still get the 9B, and `--model 9b` puts a 24 GB box back on the small model.
- **`--model` replaces reaching for an env var to change model.** Takes `35b`, `9b`,
  `auto`, or any Hugging Face repo id / local model dir; works on `chad` and `chad serve`.
  `CHAD_MODEL` still works and the flag outranks it, so a globally exported var can't pin
  every run, and `--model auto` forces the RAM-aware pick even when the env var is set.
  Forcing `35b` on a box whose RAM is too small (or unreadable) is honored but warns
  first — chad advises, you decide.
- **`serve`, `prove` and `levers` are real subcommands** with their own `--help`. They
  were dispatched on the literal task string, so `chad --help` never mentioned them while
  still listing `--host`/`--port` as "`chad serve` only" — flags that parsed fine and were
  then silently ignored on every other invocation. `--host`/`--port` now belong to
  `chad serve` and are rejected elsewhere. `chad --levers` still works as an alias.
- **`chad --help` lists 12 flags instead of 19.** The unattended-run governor knobs
  (`--think-ceiling`, `--turn-budget-tokens`, `--turn-budget-s`, `--auto-continue`,
  `--review-pass`) are set by benchmark harnesses, never by hand, and every one already
  had a `CHAD_*` twin, so the flags are **removed** rather than hidden — passing one is now
  an error instead of a silent no-op. `CHAD_AUTO_CONTINUE` fills in the one twin that was
  missing, and the bundled TB2 adapter (`benchmarks/tb2/harbor_chad_tb2.py`) sets all three
  it used through the `CHAD_*` env dict it was already building. `--think-budget` stays on
  the CLI: unlike the rest it is a capability trade a person reaches for interactively.
- **A dead remote backend explains itself.** An unreachable `--base-url` exited through a
  22-line traceback ending in `chad.base_engine.BackendError`, which reads as a chad crash
  rather than "the server you pointed me at isn't up". Now problem/cause/fix, like the
  model-load and download failures.
- **Removed the draft-model code path.** chad has been single-model since before 1.0;
  `Engine(draft_id=...)` was never set by any caller, but its branches still shaped cache
  construction, KV-quant resolution, warm-prefix eligibility and prefill. Prompt-lookup
  decoding is unaffected — it shares the word "draft" and none of the machinery.
- **`chad serve` — drive the local MLX model from a Linux container.** chad's remote arm
  (`--backend llama`) exists because MLX can't run in a benchmark container, which in
  practice meant measuring a *different quantization* of the model on a remote GPU box.
  `chad serve` exposes the in-process MLX engine — real prefix cache and all — over the
  same llama.cpp `/completion` protocol that arm already speaks, so the container drives
  the model people actually run, unchanged and with exact server-side timings.
  `--host`/`--port` (`CHAD_SERVE_HOST`/`CHAD_SERVE_PORT`), loopback and unauthenticated by
  default; `CHAD_SERVE_API_KEY` guards a wider bind. One KV cache means one agent at a
  time; concurrent clients queue.
- **Cache quarantine and warm-prefix come back over the wire.** Against a stock llama.cpp
  server these are unavoidable no-ops (its cache is opaque and its disk isn't ours). A
  `chad serve` server advertises both in `/props` and the client feature-detects, so a
  sub-agent no longer evicts the main transcript's prefix and the stable system+tools
  prefix warm-starts from disk. Nothing changes against a real llama-server, and a failed
  call degrades to the old no-op — these are latency, never correctness.
- **A client that stops reading can no longer take the server down.** Streaming wrote
  each token to the socket from the one thread allowed to touch MLX, with no timeout on
  it — so a client that stopped draining without disconnecting (a suspended container, a
  slept laptop) blocked that thread indefinitely and no other request could be served,
  while `/health` still answered `ok`. A blocked write never raises, so the hung-up-client
  cancel couldn't fire either. Generation now hands SSE lines to the request's own thread
  through a bounded queue: the engine runs ahead of a slow reader instead of being paced
  by it, and a peer that stops reading is written off the same way a disconnected one is.
- **A request no longer silently zeroes the server's sampler config.** `min_p`, `top_p`
  and `temperature` absent from a `/completion` body were defaulted to 0.0 and applied
  anyway, wiping a server started with `CHAD_MIN_P`/`CHAD_TOP_P` on essentially every
  request (the client only sends those knobs when they're armed) and forcing greedy
  decoding. Absent now means "leave it alone"; an explicit `0.0` still disarms a knob for
  that request only.
- **Generated token ids on the wire are the engine's own.** They were recovered by
  slicing the engine's cache past the prompt, which only holds on the main decode path —
  the prompt-lookup path came up one token short every turn, and after a Metal OOM the
  client was told a prefix was resident when the cache had been dropped. The engine now
  reports what it generated, and says so when it reset the cache, so the client's mirror
  and its prefill estimates stay honest on every path.
- **Long turns stopped paying quadratic CPU to watch for stop markers.** The remote
  backend re-scanned the entire generated text after every token; at 16k tokens that was
  846ms of pure scanning on the thread draining the stream. Now incremental: 4.5ms.
- **A long cold prefill no longer kills the turn it is working on.** The stream stayed
  byte-silent from the response headers until the first token, and HTTP clients measure a
  stall in bytes, not in your progress — chad's own passes its timeout to `urlopen`, which
  applies it per read. A big prefill therefore timed out just before the first token, and
  because a socket timeout looks transient the agent re-issued the step and paid for the
  same prefill again. The stream now sends an SSE comment during silence; stock clients
  discard it, so nothing on the wire changes for llama.cpp.
- **A prompt that cannot fit is a 400, not a Metal OOM.** Nothing compared the prompt
  against the context window before prefilling it, and `n_predict` was never clamped to
  the room left beside the prompt — so a client that mis-budgeted (a stale pinned context,
  a plain `curl`) walked the engine into an allocation it might not survive, taking every
  other client with it. `/completion` and `/warm` now refuse an oversized prompt up front
  and clamp the generation budget to what the window can hold. The admission wall is
  **live**, not the number pinned at load: the Metal budget is blind to other processes,
  so a docker stack or a second model started afterwards takes physical pages the KV cache
  needs 1:1, and a prompt that fit an hour ago no longer does. `/props` keeps advertising
  the stable window clients budget against; `/health` reports `safe_ctx` and
  `ctx_pressure` so a tightened wall is visible before a request bounces off it.
- **`chad serve` honors `CHAD_TEMP` / `CHAD_MIN_P` / `CHAD_TOP_P`.** It built its own
  engine and read none of them, so a server started with a sampler config sampled as if
  it had none — the one drift the server exists to eliminate. The three now travel as a
  single shared call the CLI and the server both make, so they cannot drift apart again.

## [1.0.5] — 2026-07-25

Completion and reasoning-budget accuracy, long-running commands that survive their
timeout, and a destructive-command guard that stops crying wolf. Each behavior change
is individually reversible via `CHAD_DISABLE`.

- **A `done` with nothing landed gets the task's own requirements back, once.**
  Previously, ending a turn with no landed-and-verified change stopped the turn
  immediately and banked a progress note — so the done-audit, which quotes the task
  statement's requirement lines and reports what actually exists on disk, never ran
  on exactly the turns that needed it most. That ending now hands off to a single
  audit bounce first, in context with the work the turn already did; a second empty
  ending stops as before. (`audit_churn_handoff`)
- **Reasoning truncated at the token cap now counts against the reasoning budget.**
  A generation that hit the raw output cap while still inside `<think>` credited
  zero reasoning tokens, so the per-turn budget was blind to the largest thinks
  there are — one full cap each — and its throttle never engaged on the turns that
  overspent the most. Those tokens are now counted. No threshold or clamp changed;
  only what the budget can see. (`capped_think_credit`)
- **A `bash` command that outruns its timeout keeps running instead of being killed.**
  It moves to the background rather than having its process group killed: output keeps
  streaming to a file named in the result, ending with an `[exit <code> at HH:MM:SS]`
  footer when it finishes, so a long install, build, or download is not thrown away
  and re-run from zero. Bounded by design — at most two at a time, an absolute
  lifetime each, and all of them terminated when chad exits or is signalled. A user
  interrupt still kills outright. (`bash_auto_background`)
- **The destructive-command guard screens catastrophes, not cleanups.** The
  recursive-delete check matched *any* absolute path, and in a headless container run
  every real path is absolute — so ordinary scoped deletes like
  `rm -rf /tmp/test-deploy` were refused, over and over, while the result claimed
  `[denied by user]` when no user was there. The screen now fires on targets whose
  loss is actually catastrophic — the filesystem root, top-level directories, home
  trees at any depth, the whole cwd or its parent, a glob that empties a top-level
  directory — and passes deeper scoped paths. It also screens every target of every
  `rm` in a compound command, where the old single-target check saw only the first.
  When it does block with nobody to ask, the result now says the guard (not a person)
  refused, and how to narrow the delete. mkfs / dd-to-device / fork-bomb / `curl|sh`
  screens are unchanged. (`scoped_destructive_guard`)
- **Unattended runs no longer strand a third of their time budget.** Extra fresh
  attempts after a stalled turn were only granted while more than half the wall
  budget remained, so a task whose turns kept hitting the step cap ended for good
  with 25-50% of its time unspent. Extras now keep flowing while more than a quarter
  remains; the absolute relaunch ceiling and the minimum-remaining-time floor for
  any relaunch are unchanged. (`late_continue_replenish`)
- **The system prompt steers final verification at the task's own stated check.** A
  model that was bounced for unverified work would usually re-verify — with a weaker
  check than the task named (file-exists for a content requirement, a hand-rolled
  probe instead of the stated command) — then finish wrong with most of its time
  unused. One prompt block now says: run the named check end-to-end, read its output,
  and treat existence as proof of nothing, with a WRONG/CORRECT pair.
  (`steer_verify_specific`)

Also fixed:

- **`chad --version` reported the wrong version.** `__version__` had drifted from the
  packaged version, so 1.0.3 and 1.0.4 both identified themselves as 1.0.2.

## [1.0.4] — 2026-07-22

Tool-result economics: three additions, each individually reversible via `CHAD_DISABLE`.

- **Edits survive typographic-punctuation drift.** A fourth edit-match rung: when
  exact, escape-normalized, and whitespace-flexible matching all miss, the edit
  retries with curly quotes, en/em dashes, ellipsis, and non-breaking spaces folded
  to ASCII on both sides — the drift when the model re-types prose or docstrings it
  saw rendered (in either direction). A unique match is still required, and the
  result discloses the recovery.
- **Duplicate read-only output is elided.** When a `read`/`grep`/`glob`/symbol-tool
  result comes back byte-identical to a result still in the transcript, chad appends
  a short pointer to the earlier copy instead of re-sending the full body — on a
  non-trimmable prefix cache every duplicate body is prefill paid again on every
  later step. Byte-equality against the live transcript is the safety proof: a
  changed file, different arguments, or a compaction rewrite all break equality, so
  content is only elided while a verbatim copy is provably still in context.
- **Sub-agent reports with no evidence are flagged.** A `task` sub-agent that
  returns a confident, non-empty report having dispatched zero tools answered from
  model memory, not this repository; its folded result now carries an explicit
  "verify with grep/read before relying on it" warning. Warn-not-reject on purpose:
  a re-spawn doubles the local GPU cost, and a wrongly-rejected report breaks the
  turn while a wrongly-accepted one merely restores the old behavior.

Plus a privacy default flip:

- **Local diagnostic traces are now opt-in (privacy-first default: off).** The
  readable `~/.chad/session.log` trace — user query, tool-call args (bash commands,
  write/edit content), and result previews — and the persistent input history at
  `~/.chad/history` are no longer written by default. chad is a local, single-user
  agent so nothing ever left the machine, but the log still landed plaintext previews
  under `~/.chad`, so it's now enabled only when you opt in with **`CHAD_SESSION_LOG=1`**.
  `CHAD_NO_SESSION_LOG=1` still works as a hard kill switch and wins over the opt-in.

## [1.0.3] — 2026-07-21

Guardrail-interaction fixes: three cases where the model was doing fine but
overlapping guardrails ended the turn early or accepted a non-answer.

- **A malformed tool call is never accepted as a final answer.** When the model
  slips into a wrong tool-call dialect, garbles now get their own re-nudge budget
  (separate from the token-cap-truncation counter), a canonical call exemplar once
  it's stuck, a scrub of the previous garbled message so the model stops
  few-shotting its own broken dialect, and — if the budget runs out — a clean
  hard stop with a banked progress note instead of shipping the garble as the
  answer. On by default; reversible via `CHAD_DISABLE`.
- **Done-audit re-bounce for still-missing deliverables.** If the audit flags a
  task-named output path as absent and it's *still* absent when the model tries to
  finish (with real time left on the budget), chad bounces once more naming only
  the missing path, then accepts unconditionally. Stops a required file from going
  unwritten while the turn ends "successfully."
- **Investigation gate exempts ops commands.** A bash step that isn't provably
  read-only (`git merge`, `apt-get install`, redirects, `make`, …) now counts as
  action, not investigation, so the "you've only been looking, make an edit" gate
  no longer fires in the middle of a legitimate ops workflow.

## [1.0.2] — 2026-07-20

Reliability tuning for long, budgeted tasks, plus one interactive papercut fix.

- **Syntax gate no longer flags prose & data files.** Plain-text deliverables — a
  `.txt` answer file, `requirements.txt`, markdown, CSV/TSV — were being run through a
  tree-sitter grammar (the language pack maps `.txt` → VIMDOC) and warned on, sometimes
  reverted, at exactly the write that produced the deliverable. The gate now polices
  code languages only; prose/data formats are skipped uniformly across warn, edit-revert,
  and write-reject.
- **`run`-style tasks no longer bail with prose.** System-state asks ("start the
  service", "boot the image", "install X") are completable with zero file edits, and
  used to fall through to the weakest completion path — a give-up in prose could ship
  nothing (qemu-startup did, with most of the wall unspent). They now get their own
  intent class that arms the anti-bail nudges, while still completing cleanly with zero
  edits (the no-empty-diff "done" gate only ever applied to edit tasks).
- **Reasoning budget throttles instead of muting.** Once a turn's cumulative
  reasoning-token budget is spent, chad now forces one no-think action step per ~3k
  further reasoning tokens (a duty cycle) rather than muting thinking for the rest of the
  turn — thinking is restored as soon as the model stops over-spending, avoiding the
  garbled tool-call tails a blanket mute produced.
- **Wall-aware auto-continue.** When most of a task's wall budget is still unspent, an
  exhausted turn is granted a fresh relaunch (bounded), instead of giving up after a
  fixed two attempts with the clock barely touched.
- **Rejected "done" no longer poisons the relaunch.** When a completion claim is
  rejected for landing no verified change, the carried-forward progress note now leads
  with a warning that the claim was rejected and drops any hypothesis that itself asserts
  completion — so a relaunch stops inheriting "already complete and verified" as fact and
  re-confirming it.

## [1.0.1] — 2026-07-19

First complete release: a single-user coding agent that runs **entirely locally on
Apple Silicon** — Claude-Code-style tool use, plan mode, and a full-screen TUI, driven
by an Ornith model picked for your RAM (35B MoE at ≥32 GB, 9B dense below). One
command: `uvx chad-code`.

### The agent

- **Full-screen TUI**: shift-tab permission modes (normal / auto-accept / plan), a
  type-ahead message queue, mid-turn interrupt, mid-run steering (keep typing while it
  works — injected between steps, no re-prefill), and a live status line (↑prefilled /
  ↓generated counts, advancing % on a full re-prefill).
- **Sessions**: resume (`-c`) and list-and-fork (`--resume`) — copy-on-resume, so a
  resume never overwrites the session it branched from.
- **Extensions**: Agent Skills ([agentskills.io](https://agentskills.io)) and MCP
  servers (stdio & Streamable HTTP; OAuth behind `CHAD_MCP_OAUTH=1`), with a
  project-trust gate — a project's `./.mcp.json` servers don't start until `/mcp trust`.
- **Sub-agents** (`task` tool): open-ended exploration runs in a quarantined context
  with a read-only toolset, so the main transcript stays small and the warm cache
  survives the spelunking.

### The engine

- **Persistent prefix KV cache** — the core design: the transcript is kept a strict
  token-prefix of the live cache, so each step prefills only what was appended. The
  stable system+tools prefix checkpoints to disk (a cold start loads it instead of
  re-prefilling); a divergence inside the last turn (a truncated generation
  re-rendered, a retried prompt) takes a **bounded rewind** — snapshot-restore the
  recurrent state, native-trim the attention KV, re-feed one turn — instead of a
  full-transcript rebuild. Bit-exactness pinned by test on the hybrid SSM/attention
  cache.
- **35B-on-24GB stability**: Metal allocator clamps installed at load; a Metal OOM
  inside a prefill chunk rolls the cache back exactly, halves the chunk and retries
  (mlx ≥ 0.32 required for catchable OOMs); adaptive per-chunk prefill sizing
  (MoE 2048 / dense 512, decaying under pressure — worth ~+14% prefill on the 35B);
  the RAM-aware compaction trigger respects host-physical pressure and re-checks per
  turn; compaction that cannot get under the window latches instead of destroying the
  warm cache every step (the old behavior cost a measured 26–28 s per step).
- **Correct sampling off the main thread**: MLX's implicit-key RNG never advances on
  non-main threads, so temp>0 decoding there was quasi-greedy with one frozen noise
  vector (identical prompts → byte-identical responses, markedly loop-prone). An
  explicit split-per-draw key chain (`_KeyedSampler`, fresh entropy per generation)
  fixes it on every thread; greedy decoding is unchanged.
- **Remote arm** (`--backend llama`): the same harness against a llama.cpp server's
  raw `/completion` — token-id prompts (no double chat-template), real cache
  telemetry, `<think>` passed back verbatim. This is the Terminal-Bench arm; see
  `benchmarks/tb2/`.

### Reliability on a small model

The bet: for a C+ student, the harness is worth more than a model upgrade. Every
lever below exists because a logged failure demanded it.

- **Tool calls parsed in four dialects** (JSON, fenced, XML function-call, and the
  hybrid garble a quantized model emits under sampling), then schema-coerced with a
  typed self-repair loop — the model is shown exactly which fields are wrong instead
  of regenerating blindly.
- **Edits go through a forgiveness cascade** (exact → escape-normalized →
  whitespace-flexible with the file's own indentation), and line-addressed
  `replace_lines`/`insert_lines` fit indentation for you. An edit that would newly
  break the file's parse is **reverted, not landed** — with visible-whitespace echoes
  so the model copies real indentation instead of re-guessing.
- **Guard rails with teeth**: loop/repetition/thrash detection with bounded nudges,
  verify-before-done (a display command exiting 0 is not verification), a done-audit
  that quotes the task's own requirements back, a no-empty-diff gate on action tasks,
  and a runaway-turn governor that banks a deterministic progress note and relaunches
  fresh instead of grinding a poisoned context.
- **Progress-aware step cap**: a turn that keeps landing *and verifying* changes earns
  extensions (ceiling 4×); one that stalls banks its progress note so
  `continue`/`--auto-continue` resumes instead of the task dying at the prompt.
- **Bounded prefill everywhere**: every tool result is capped before it hits the
  transcript, and a multi-call step shares one output budget — a whole-file read or
  wide grep can no longer stall the next turn 20–30 s.
- **Degenerate-repetition stop**: decode locking into repeating one short string
  (minutes of dead generation per occurrence) is detected within a few hundred
  tokens; the step is cut off and the model nudged out of the loop.

### The models

- **35B `UD-Q2_K_XL` quant improved at the same speed**: 3-bit expert
  down-projections (3.09 bits/weight, ~13.4 GB resident); held-out code perplexity
  7.25 → 6.86 with prefill/decode unchanged — decode reads only the 8 active experts,
  so the extra bits are effectively free.
- **A 6-bit build for big-memory Macs**:
  [`Ornith-1.0-35B-Q6-MLX`](https://huggingface.co/nathansutton/Ornith-1.0-35B-Q6-MLX)
  (~28.5 GB; needs ≥48 GB) — run it with `CHAD_MODEL=<repo>`.

### Safety & first run

- Catastrophic-bash seatbelt (screened even in `--yolo`; blocked headless),
  deterministic post-edit syntax gate, minimal environment for stdio MCP subprocesses,
  sessions stored `0600`, secret-redacted diagnostic log.
- **`chad prove`**: a two-minute offline smoke test — four tiny fix-it tasks through
  the real agent loop, mechanically verified, network-blocked, timed. A proof of
  life, not a benchmark.
- Disk preflight before the model download (refuses with the shortfall instead of
  dying at 70%); unknown-RAM machines get the safe 9B; guard stops say what to do
  next and point at the troubleshooting symptom map.

### Benchmarks & docs

- **Terminal-Bench reproduction kit** (`benchmarks/tb2/`): the exact Harbor adapter,
  runners (including the leaderboard-compliant `run_tb21_submit.sh`), and the ATIF
  trajectory validator behind the README chart — serve an Ornith GGUF with llama.cpp
  yourself and check the number rather than trust it.
- Design & internals, configuration reference, throughput docs (reproduce with
  `chad-bench`), troubleshooting symptom map, release checklist (`RELEASING.md`),
  and a demo GIF regenerated from a checked-in vhs tape so it can't rot silently.
