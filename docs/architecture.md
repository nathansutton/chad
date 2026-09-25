# Architecture

*Reference for contributors: what each module owns, which tests guard it, and the two
on-disk/on-wire formats. Why the pieces are shaped this way is in [design](design.md).*

```
cli.py ──▶ agent.py (agentic loop + guardrails) ──▶ engine.py (MLX + persistent prefix cache)
                 │
                 ├─ tools.py (bash · edit · write · write_todos · done)
                 └─ ambient.py (what the result channel adds back)
```

## Architecture map

Every module in `src/chad/`, what it owns, and the tests that guard it. The map rots one
row at a time, so a PR that adds or deletes a module should add or delete its row. Test
files are listed by which ones import the module (plus, where noted, the ones that drive
it through a re-export).

| Module | Responsibility | Guarding tests |
|---|---|---|
| `__init__.py` | Package docstring and `__version__` — the string `--version` prints and the ATIF trajectory records; must match `pyproject.toml`. | `test_cli.py` |
| `agent.py` | The agentic loop and REPL: render the transcript through the chat template, stream the turn, parse tool calls, run them, feed results back, repeat. | `test_agent.py`, `test_agent_guards.py`, `test_agent_e2e.py`, `test_intent.py` |
| `ambient.py` | Ambient state for the result channel: the levers that append harness knowledge to results the model already reads, rather than adding tools. | `test_ambient.py`, `conftest.py` |
| `atif.py` | ATIF v1.7 trajectory emitter, rebuilt from `agent.messages` after each step, with each step's cost in `metrics.extra` and the step still being generated as a trailing `in_flight` step; a pure observer armed by `CHAD_TRAJECTORY_JSON`, or per trial by `atif.start`. | `test_atif.py` |
| `base_engine.py` | The engine seam: `GenStats` plus the `BaseEngine` Protocol that `Agent` already drives, so a second backend plugs in without touching the agent loop. | `test_completion_engine.py`, `test_agent_e2e.py`, `test_cli.py` |
| `bench.py` | The throughput benchmark behind `docs/benchmarks.md` (`chad-bench`): cold prefill, decode and warm-step tok/s on the real engine and public model. | `test_bench.py` |
| `checkpoint.py` | Shadow-git snapshots of the workspace before each file-mutating tool, in their own GIT_DIR, so `/undo` and `/restore` can revert an auto-approved edit. | `test_checkpoint.py` |
| `cli.py` | Argument parsing and entrypoint (`chad.cli:main`), plus the `prove` and `levers` subcommands. | `test_cli.py`, `test_cli_modes.py` |
| `compaction.py` | Context compaction for long sessions: every pass marks what it trimmed in-band and the notice names the spill file holding the original. | `test_compaction.py`, `test_skills.py` |
| `completion_engine.py` | The one remote backend (`--backend llama`): llama.cpp's native `/completion` endpoint driven with token-id prompts and real cache telemetry. | `test_completion_engine.py` |
| `config.py` | Single source of truth for `CHAD_*` configuration — typed accessors whose lenient parse warns and degrades a bad value to the default instead of raising. | `test_config.py`, `test_seatbelt.py` |
| `diag.py` | The opt-in diagnostic session log (`CHAD_SESSION_LOG`): throughput numbers, tool args and result previews, secret-redacted and size-rotated, never model-facing. | `test_log_redaction.py` |
| `engine.py` | The MLX inference engine and its persistent prefix KV cache, extended across turns by diffing token ids so only the newly appended tokens prefill. | `test_engine.py`, `test_engine_dflash.py`, `test_engine_kvquant.py`, `test_engine_pld_hybrid.py`, `test_engine_pld_wide.py` |
| `gguf_pack.py` | Loader for Unsloth's GGUF checkpoints of the qwen3_5 hybrid: derives the model directory from the GGUF header, undoes llama.cpp's converter layouts, and keeps every projection in its original blocks. | `test_gguf_pack.py`, `test_cli.py` |
| `guardrails.py` | The pure decision predicates `run_turn` calls: loop guard, verify-before-done and empty-done gating, tool-result bookkeeping, no-tool-call nudge selection. | `test_agent_guards.py`, `test_gate.py`, `test_agent_e2e.py` |
| `ignore.py` | Single source of truth for directories no tree-walk enters (`IGNORE_DIRS`, plus `REPOMAP_EXTRA` for the repo-analysis path). | `test_ignore.py` (through the `tools`/`repomap`/`skills` re-exports) |
| `levers.py` | The registry of shipped bash-route levers, all ON by default, each keeping a name and an `enabled()` guard so `CHAD_DISABLE=a,b` can ablate one at a time. | `test_levers.py`, `test_ambient.py` |
| `mcp.py` | MCP client: read `.mcp.json`/`~/.chad/mcp.json`, connect each server over the SDK transport, expose its tools as `mcp__<server>__<tool>`, dispatch calls. | `test_mcp.py`, `test_mcp_oauth.py` |
| `mcp_oauth.py` | OAuth for hosted HTTP MCP servers: 0600 per-server token storage plus the browser/loopback redirect flow, behind the `CHAD_MCP_OAUTH` flag. | `test_mcp_oauth.py`, `test_mcp.py` |
| `mlx_dflash.py` | The DFlash2 block-diffusion drafter — load, forward, and the target-residual tap — that proposes a whole block of tokens in one drafter forward. | `test_engine_dflash.py`, `test_cli.py` |
| `mlx_fastpath.py` | Decode fast path for the dense qwen3_5 hybrid: per-row-exact weight concats and compiled S==1 layer steps that remove dispatch-bound kernel launches. | `test_mlx_fastpath.py`, `test_engine_dflash.py` |
| `mlx_gguf.py` | GGUF block dequantizers as Metal device functions (K-quants, i-quants, IQ4), bit-exact to gguf.quants, under a fused GEMV, an MMA verify kernel and a sliced expand for prefill. | `test_gguf_kernels.py`, `test_gguf_pack.py` |
| `mlx_qmm_mma.py` | Small-M quantized matmul for speculative verify: an MMA kernel that reads each weight group once for all rows, where stock GEMV re-pays the read per row. | `test_mlx_qmm_mma.py` |
| `mlx_qsdpa.py` | JIT-compiled fused attention over the 8-bit group-64 quantized KV cache, serving decode, speculative verification and prefill. | `test_mlx_qsdpa.py`, `test_engine_kvquant.py` |
| `prompt.py` | System-prompt construction (static prompt, workspace snapshot, project instructions) and the answer-on-paper / verify-nudge intent classifier. | `test_intent.py`, `test_warm_prefix_tiers.py`, `test_ambient.py`, `test_skills.py` |
| `prove.py` | `chad prove` — a two-minute smoke test pinned to the shipped model, offline-guarded after the cache check, reporting your own machine's numbers. | `test_prove.py` |
| `render.py` | Terminal rendering: raw tokens and tool results into a compact activity view, behind the `_emit(kind, text)` callback the REPL and the TUI each supply. | `test_render.py`, `test_confirm_preview.py`, `test_feel_pack.py` |
| `repomap.py` | Tree-sitter tag extraction for the ambient levers: language detection, mtime-cached per-file definitions, and cross-file definition lookup. | `test_repomap.py`, `test_repomap_polyglot.py`, `test_ambient.py` |
| `seatbelt.py` | macOS Seatbelt confinement for yolo-mode bash: the spawned shell (never the chad process, which needs Metal) is denied writes outside the workspace. | `test_seatbelt.py` |
| `session.py` | Conversation persistence per project directory, so `--continue`, `--resume` and the TUI `/resume` picker survive across runs. | `test_session.py`, `test_cli_modes.py`, `conftest.py` |
| `skills.py` | Agent Skills: discover `SKILL.md` dirs, parse frontmatter leniently, offer each as a slash command, and load only the one the user asks for as a user turn. | `test_skills.py`, `test_validate.py`, `test_ignore.py` |
| `speech.py` | All-local speech I/O for the TUI — Parakeet-on-MLX dictation and macOS `say` replies — with the heavy audio/MLX imports deferred to first use. | `test_speech.py`, `test_speech_tui.py` |
| `spill.py` | Spill files: every truncation writes the dropped body to disk first and the notice names the path, so a clip is a loan rather than a deletion. | `test_spill.py`, `test_compaction.py`, `test_intent.py` |
| `syntaxgate.py` | The post-mutation syntax warning: a landed edit/write that *newly* breaks a file's parse says so in the same result the model is about to read. | `test_syntaxgate.py` (through `tools.tool_edit`/`tool_write`) |
| `toolcall_parse.py` | The pure boundary between raw model text and tool dispatch: parse every tool-call dialect local models emit, de-duplicated. | `test_toolcall_parse.py`, `test_toolcall_dialect.py` |
| `tools.py` | The tool surface — `bash`, `edit`, `write`, `write_todos`, `done` — with its JSON schemas and the edit forgiveness cascade. | `test_tools.py`, `test_edit.py`, `test_gate.py`, `test_syntaxgate.py` |
| `tui.py` | The full-screen prompt_toolkit UI: mode cycling, type-ahead queue, interrupt, inline approval and status line, printed into the terminal's normal scrollback. | `test_tui.py`, `test_feel_pack.py`, `test_render.py`, `test_speech_tui.py` |
| `validate.py` | Typed tool-call validation and self-repair over `tools.SCHEMAS`: lenient parse, coerce, validate, then feedback naming exactly which fields were wrong. | `test_validate.py`, `test_tools.py`, `test_mcp.py`, `test_toolcall_dialect.py` |

## Session file format

```
~/.chad/sessions/<cwdhash>/<session_id>.json   one file per session
~/.chad/sessions/<cwdhash>/index.json          {title, updated, turns} per session id
```

A session file is a JSON object with five keys: `cwd` (absolute), `session_id`
(`YYYYMMDD-HHMMSS-<4 hex>`, minted at `Agent` construction), `updated` (epoch seconds),
`meta`, and `messages`. Only the message list is persisted — never the KV cache — so a
resume re-prefills the restored transcript. Each save overwrites only its own session file
and refreshes that session's `index.json` entry, so the `/resume` picker lists a directory
without opening every session, and resuming (which mints a fresh id and seeds the old
messages) is implicitly a fork. Known-prefix secrets are masked in tool results and
tool-call arguments on the way to disk; the in-memory transcript keeps the real values. A
legacy single-slot `~/.chad/sessions/<cwdhash>.json` is adopted as one session the first
time that directory is listed. Every write is best-effort: failing to save never breaks a
turn.

## Tool-call wire format

`toolcall_parse.parse_tool_calls` strips `<think>` blocks, then tries these dialects in
order and de-duplicates what it finds. The model's own dialect — what the chat template
renders and what the tools are described in — is the `<tool_call>` wrapper; the rest exist
because weaker local models reach for the others.

1. The XML function-call dialect `<function=name><parameter=key>value</parameter></function>` (Qwen3 / GLM thinking models); it wins when present.
2. The hybrid `{"name": "x"}` followed by `<parameter=…>` blocks — only when real parameter blocks are in scope, so a plain JSON call is left to the JSON path.
3. `<tool_call>{"name": …, "arguments": {…}}</tool_call>`.
4. A fenced block: ` ```json ` or ` ```tool_call `.
5. If nothing above matched, a closed `<tool_call>` whose JSON never closed (XML cruft stripped), and finally any bare top-level JSON object in the text.

A body that will not parse is repaired rather than dropped, because a dropped call leaves
the model with no result and no idea why: `validate.repair_json` strips the fence, fixes
trailing commas, Python constants and bare keys outside string literals, then closes
unterminated strings and brackets. A whole `arguments` value double-encoded as a string is
re-parsed the same way. What survives goes through `validate.coerce_and_validate` against
the same `tools.SCHEMAS` dict the model was sent, and any errors become
`validate.render_repair` feedback that annotates the model's own arguments, so it repairs
the marked fields instead of regenerating blindly.
