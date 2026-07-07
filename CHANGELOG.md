# Changelog

Notable, user-visible changes. Started 2026-07; earlier history is summarized coarsely.

## [Unreleased]

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
