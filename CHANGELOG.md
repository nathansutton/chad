# Changelog

Notable, user-visible changes. Started 2026-07; earlier history is summarized coarsely.

## [Unreleased]

- **`chad prove` — a two-minute offline smoke test**: four tiny fix-it tasks from
  chad's own dev suite (disclosed as such — it's a proof of life, not a
  benchmark), run through the real agent loop on the validated 9B, mechanically
  verified with tamper-proof re-seeded checkers, and timed. Scorecard reports
  time-to-first-token, decode tok/s, and per-task wall; network access is
  blocked at the library level during task execution; a share snippet is offered
  only on a full pass. Exit codes: 0 all pass / 1 any fail / 2 preflight stop.
  The task set survived a 3-condition falsification gate (3/3 reps under 60 s,
  including memory-constrained) before this shipped.
- **Disk preflight before the model download**: chad now checks free space at the
  Hugging Face cache before offering the 5–12 GB download and refuses with the
  shortfall and the `hf cache rm` pointer instead of dying at 70%; a mid-download
  disk-full failure is now diagnosed as such. The confirm prompt also gained a
  while-you-wait tip (cd into a project, think of a scoped first ask).
- **Unknown-RAM machines get the safe model**: if RAM can't be detected, chad now
  picks the 9B instead of silently falling through to the 35B's 12 GB download.
- **Guard stops now say what to do next**: the loop, repetition, and step-cap
  aborts carry a recovery tip and point at the troubleshooting symptom map
  instead of ending the turn with only a diagnosis.
- **README restructured**: the one-line quickstart now sits at the top, the
  thesis prose after it, and all other install paths under a new
  "Installing & upgrading" section; new
  [Why there's no model picker](docs/design.md#why-theres-no-model-picker)
  section in the design docs.
- **Troubleshooting**: new symptom row for the silent mid-turn exit (an MLX
  Metal abort under memory pressure prints nothing by nature — the row says
  which knob helps and where the crash report lands).
- **Community & releasing**: GitHub Discussions enabled (Q&A + Show and tell),
  issue templates point questions there; `RELEASING.md` documents the release
  checklist; the README demo GIF is regenerated from a checked-in vhs tape
  (`docs/demo.tape`) so the demo can't rot silently.
- **Progress-aware step cap**: a turn that keeps landing *and verifying* changes is no
  longer killed dead at 40 tool steps mid-task — it earns +20-step extensions (absolute
  ceiling 4×) on the warm cache, matching the governor's never-interrupt-real-progress
  rule. A turn that does stop at the cap now banks a progress note (same contract as a
  governor hard-stop), so `continue` / `--auto-continue` resume it instead of the task
  silently dying at the prompt.
- **Responsiveness**: every tool result is bounded before it hits the transcript
  (whole-file reads, wide greps, and the symbolic map tools were each able to stall the
  next turn 20–30s on prefill), and a step that makes several tool calls now shares one
  per-step output budget so the results can't stack into a single giant prefill.
- **Degenerate-repetition stop**: greedy decode locking into repeating one short string
  (minutes of dead generation per occurrence, the main task-timeout cause in dogfooding)
  is now detected within a few hundred tokens; the step is cut off and the model nudged
  out of the loop. `CHAD_NO_REPEAT_GUARD=1` disables (A/B knob).

## [0.1.0] — 2026-07

First coherent public cut. Highlights since `v0.0.1`:

- **Full-screen TUI**: shift-tab permission modes (normal / auto-accept / plan), a
  type-ahead message queue, mid-turn interrupt, and a live status line (↑prefilled /
  ↓generated token counts, ✂N adaptive think-cap counter).
- **Sessions**: resume (`-c`) and list-and-fork (`--resume`, `/resume`) — copy-on-resume,
  so a resume never overwrites the session it branched from.
- **Extensions**: Agent Skills ([agentskills.io](https://agentskills.io)) and MCP servers
  (stdio & Streamable HTTP; OAuth behind `CHAD_MCP_OAUTH=1`), with a project-trust gate —
  a project's `./.mcp.json` servers don't start until you `/mcp trust` them.
- **Engine**: persistent prefix KV cache with warm-prefix disk checkpoints, RAM-aware
  context sizing, an adaptive think-cap (`--think-budget`), and a runaway-turn governor.
- **Safety**: catastrophic-bash seatbelt, deterministic post-edit syntax gate, and a
  minimal environment for stdio MCP subprocesses.
- **Fixed**: `transformers` capped `<5.13` to fix an mlx-lm import crash on fresh installs.
