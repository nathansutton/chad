# Changelog

Notable, user-visible changes.

## [1.0.0] — 2026-07-19

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
