# Architecture

What each module owns, which tests guard it, and the two formats on disk and on the wire.
Why the pieces are shaped this way is in [design](design.md).

```
cli.py ──▶ agent.py (agentic loop + guardrails) ──▶ engine.py (MLX + persistent prefix cache)
                 │
                 ├─ tools.py (bash · edit · write · write_todos · done)
                 └─ ambient.py (what the result channel adds back)
```

## Architecture map

Every module in `src/chad/`. A PR that adds or deletes a module adds or deletes its row.
Tests are listed by which ones import the module.

| Module | Responsibility | Guarding tests |
|---|---|---|
| `__init__.py` | `__version__`, which `--version` prints and the ATIF trajectory records; must match `pyproject.toml`. | `test_cli.py` |
| `agent.py` | The agentic loop and REPL: render the transcript, stream the turn, parse tool calls, run them, feed results back, repeat. | `test_agent.py`, `test_agent_guards.py`, `test_agent_e2e.py`, `test_intent.py`, `test_kv_resume.py` |
| `ambient.py` | The result-channel levers: harness knowledge appended to results the model already reads; its facts are saved with the session. | `test_ambient.py`, `test_ambient_resume.py`, `conftest.py` |
| `atif.py` | ATIF v1.7 trajectory emitter, a pure observer armed by `CHAD_TRAJECTORY_JSON`. | `test_atif.py` |
| `base_engine.py` | The engine seam: `GenStats`, `KVCheckpointRef`, and the `BaseEngine` Protocol the agent drives, so a second backend plugs in without touching the loop. | `test_completion_engine.py`, `test_agent_e2e.py`, `test_cli.py`, `test_kv_resume.py` |
| `bench.py` | `chad-bench`: cold prefill, decode and warm-step tok/s on the real engine. | `test_bench.py` |
| `checkpoint.py` | Shadow-git snapshots of the workspace before each mutating tool, so `/undo` and `/restore` can revert. | `test_checkpoint.py` |
| `cli.py` | Argument parsing, the entrypoint, and the `prove` and `levers` subcommands. | `test_cli.py`, `test_cli_modes.py` |
| `compaction.py` | Context compaction: every pass marks what it trimmed in-band and names the spill file holding the original. | `test_compaction.py`, `test_skills.py` |
| `completion_engine.py` | The one remote backend (`--backend llama`): llama.cpp's `/completion` endpoint with token-id prompts and real cache telemetry. | `test_completion_engine.py` |
| `config.py` | Single source of truth for `CHAD_*` configuration: typed accessors that warn and degrade a bad value to the default. | `test_config.py`, `test_seatbelt.py` |
| `diag.py` | The opt-in session log (`CHAD_SESSION_LOG`): secret-redacted, size-rotated, never model-facing. | `test_log_redaction.py` |
| `engine.py` | The MLX engine and its persistent prefix KV cache, extended across turns by diffing token ids; checkpointed to disk for the stable prefix and for the whole session at quit. | `test_engine.py`, `test_engine_dflash.py`, `test_engine_kvquant.py`, `test_engine_pld_hybrid.py`, `test_engine_pld_wide.py`, `test_warm_prefix_tiers.py` |
| `gguf_pack.py` | Loader for Unsloth's GGUF checkpoints of the qwen3_5 hybrid: undoes llama.cpp's converter layouts and keeps every projection in its original blocks. | `test_gguf_pack.py`, `test_cli.py` |
| `guardrails.py` | The pure predicates `run_turn` calls: loop guard, verify-before-done and empty-done gating, no-tool-call nudges. | `test_agent_guards.py`, `test_gate.py`, `test_agent_e2e.py` |
| `ignore.py` | The directories no tree-walk enters. | `test_ignore.py` (through the `tools`/`repomap`/`skills` re-exports) |
| `levers.py` | The registry of result-channel levers, each with a name and an `enabled()` guard so `CHAD_DISABLE` can ablate one at a time. | `test_levers.py`, `test_ambient.py` |
| `mcp.py` | MCP client: read the config files, connect each server, expose its tools as `mcp__<server>__<tool>`, dispatch calls. | `test_mcp.py`, `test_mcp_oauth.py` |
| `mcp_oauth.py` | OAuth for hosted MCP servers: per-server token storage and the browser/loopback redirect flow, behind `CHAD_MCP_OAUTH`. | `test_mcp_oauth.py`, `test_mcp.py` |
| `mlx_dflash.py` | The DFlash2 block drafter: load, forward, and the target-residual tap. | `test_engine_dflash.py`, `test_cli.py` |
| `mlx_fastpath.py` | Decode fast path for the dense qwen3_5 hybrid: fused projections and compiled S=1 layer steps. | `test_mlx_fastpath.py`, `test_engine_dflash.py` |
| `mlx_gguf.py` | GGUF block dequantizers as Metal device functions, bit-exact to gguf.quants, under a fused GEMV, an MMA verify kernel and a sliced expand for prefill. | `test_gguf_kernels.py`, `test_gguf_pack.py` |
| `mlx_qmm_mma.py` | Small-M quantized matmul for speculative verify: reads each weight group once for all rows. | `test_mlx_qmm_mma.py` |
| `mlx_qsdpa.py` | Fused attention over the 8-bit quantized KV cache, for decode, verify and prefill. | `test_mlx_qsdpa.py`, `test_engine_kvquant.py` |
| `prism_pack.py` | Loader for Hadamard-rotated 2-bit packs, which mlx-lm's plain loader would load without the rotation. | `test_prism_pack.py` |
| `prompt.py` | System-prompt construction and the answer-on-paper / verify-nudge intent classifier. | `test_intent.py`, `test_warm_prefix_tiers.py`, `test_ambient.py`, `test_skills.py` |
| `prove.py` | `chad prove`: a smoke test pinned to this machine's default model, offline after the cache check. | `test_prove.py` |
| `render.py` | Terminal rendering of tokens and tool results, behind the `_emit(kind, text)` callback the REPL and TUI supply. | `test_render.py`, `test_confirm_preview.py`, `test_feel_pack.py` |
| `repomap.py` | Tree-sitter tag extraction for the ambient levers, mtime-cached per file. | `test_repomap.py`, `test_repomap_polyglot.py`, `test_ambient.py` |
| `seatbelt.py` | macOS Seatbelt confinement for yolo-mode bash: the spawned shell, never the chad process. | `test_seatbelt.py` |
| `serve.py` | `chad-serve`: the engine behind an OpenAI-compatible `/v1/chat/completions`, for other agents (see `docs/pi.md`). | `test_serve.py` |
| `session.py` | Conversation persistence per project directory, with the ref to the session's KV checkpoint when it ended cleanly. | `test_session.py`, `test_cli_modes.py`, `test_kv_resume.py`, `conftest.py` |
| `skills.py` | Agent Skills: discover `SKILL.md` dirs, parse frontmatter leniently, offer each as a slash command, load only the one asked for. | `test_skills.py`, `test_validate.py`, `test_ignore.py` |
| `slash.py` | The check both front ends run before a line reaches the model: an unknown slash command is reported, not sent as a task. | `test_slash.py`, `test_tui.py` |
| `speech.py` | All-local speech I/O: Parakeet-on-MLX dictation and macOS `say`, imports deferred to first use. | `test_speech.py`, `test_speech_tui.py` |
| `spill.py` | Spill files: every truncation writes the dropped body to disk first and the notice names the path. | `test_spill.py`, `test_compaction.py`, `test_intent.py` |
| `suffix_reuse.py` | After compaction, move the attention rows of surviving text to their new positions instead of re-reading it. | `test_suffix_reuse.py`, `test_engine.py` |
| `syntaxgate.py` | The post-edit syntax warning: an edit that newly breaks a file's parse says so in the same result. | `test_syntaxgate.py` (through `tools.tool_edit`/`tool_write`) |
| `toolcall_parse.py` | The boundary between raw model text and tool dispatch: every tool-call dialect local models emit, de-duplicated. | `test_toolcall_parse.py`, `test_toolcall_dialect.py` |
| `tools.py` | The tool surface (`bash`, `edit`, `write`, `write_todos`, `done`), its JSON schemas and the edit forgiveness cascade. | `test_tools.py`, `test_edit.py`, `test_gate.py`, `test_syntaxgate.py` |
| `tui.py` | The prompt_toolkit UI: mode cycling, type-ahead queue, interrupt, inline approval and status line. | `test_tui.py`, `test_feel_pack.py`, `test_render.py`, `test_speech_tui.py` |
| `validate.py` | Tool-call validation and self-repair over `tools.SCHEMAS`: lenient parse, coerce, validate, then feedback naming the wrong fields. | `test_validate.py`, `test_tools.py`, `test_mcp.py`, `test_toolcall_dialect.py` |

## Session file format

```
~/.chad/sessions/<cwdhash>/<session_id>.json   one file per session
~/.chad/sessions/<cwdhash>/index.json          {title, updated, turns} per session id
```

A session file is a JSON object with five keys: `cwd`, `session_id`
(`YYYYMMDD-HHMMSS-<4 hex>`), `updated` (epoch seconds), `meta`, and `messages`.

- **`meta.kv`** is a `KVCheckpointRef` (`{path, tokens, sha}`): a session that ends cleanly
  writes the engine's cache to `~/.cache/chad/kv/sess-<hash>.safetensors` and records the
  first `tokens` ids of the transcript and their sha1. A resume whose transcript still
  begins with those ids restores the file and prefills only what is new; anything else
  re-prefills. The filename hash also covers the model, cache mode and window.
- **`meta.ambient`** is `ambient.snapshot()`: the files the session edited, its pre-edit
  test baseline, and the files it was shown a skeleton of, restored into the resumed agent.
- Each save overwrites only its own file and refreshes its `index.json` entry. Resuming
  mints a fresh id and seeds the old messages, so a resume is implicitly a fork.
- Known-prefix secrets are masked on the way to disk; the in-memory transcript keeps the
  real values. Every write is best-effort: failing to save never breaks a turn.

## Tool-call wire format

`toolcall_parse.parse_tool_calls` strips `<think>` blocks, then tries these dialects in
order and de-duplicates. The model's own dialect is the `<tool_call>` wrapper; the rest
exist because weaker local models reach for them.

1. `<function=name><parameter=key>value</parameter></function>` (Qwen3 / GLM thinking
   models); it wins when present.
2. `{"name": "x"}` followed by `<parameter=…>` blocks, only when real parameter blocks are
   in scope.
3. `<tool_call>{"name": …, "arguments": {…}}</tool_call>`.
4. A fenced ` ```json ` or ` ```tool_call ` block.
5. A closed `<tool_call>` whose JSON never closed, then any bare top-level JSON object.

A body that will not parse is repaired rather than dropped, because a dropped call leaves
the model with no result and no idea why: `validate.repair_json` strips fences, fixes
trailing commas, Python constants and bare keys, and closes unterminated strings and
brackets. What survives is validated against the same `tools.SCHEMAS` the model was sent,
and errors come back as `validate.render_repair` feedback annotating the model's own
arguments, so it repairs the marked fields instead of regenerating blindly.
