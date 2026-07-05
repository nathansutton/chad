# Changelog

Notable, user-visible changes. Started 2026-07; earlier history is summarized coarsely.

## [Unreleased]

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
