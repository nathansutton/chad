# Troubleshooting

A small model on a laptop can ramble, loop, or slow down in ways a frontier API will not.
Most of it is **memory**, **scope**, or **something chad deliberately did not auto-enable**.
This page maps what you see to the knob; the knobs themselves are in the
[configuration reference](configuration.md).

**Start here if something is broadly wrong:** `chad prove` runs four tiny fix-it tasks
against the shipped model, offline, and reports what worked. It separates "my install is
broken" from "the model struggled with my task" in about a minute.

## Slow, or won't stop

| You see | What's happening | Reach for |
|---|---|---|
| Long `<think>` before every action | reasoning is on by default and is ~65% of what the model generates | `--no-think` for well-scoped work. To trim only the rambles, `CHAD_THINK_CEILING=384` force-closes a runaway block and keeps going in the same step; `--think-budget 1500` is the blunter version that ends the step ([turn budgets](configuration.md#turn-budgets--think-cap)) |
| A turn burns minutes without landing an edit | runaway turn | the governor ends it once its token budget is spent; before that, ctrl-c and re-scope the ask smaller |
| The same `sed -n` / `rg` over and over | the loop guard will abort the turn; the ask is probably too vague | a smaller, concrete task; name the file |
| First turn in a **new** project takes over a minute | the system+tools prefix is built once per distinct prompt, then cached on disk | expected once; the banner's `[warm start: …]` line says which turn you are having ([benchmarks](benchmarks.md#the-second-session-in-a-project-starts-warm)) |
| First token slow after `/compact` | compaction re-reads what it inserted; the status line shows an advancing % | expected ([design](design.md#the-cache-only-appends)) |

## Memory

| You see | What's happening | Reach for |
|---|---|---|
| Mac swap-storming with other apps open | ~14 GB of weights and drafter plus a ~2 GB prefill transient plus your apps exceed free RAM | close apps, or lower `CHAD_CTX_SAFETY` to buy headroom from the context window ([context window](configuration.md#context-window-agentic-coding-needs-room)) |
| The banner reports less context than 262k | the governor sizes the window from live free memory; 24 GB lands around ~74k | nothing: the banner states what you got. `--model` an `UD-IQ3_XXS` GGUF for ~138k |
| chad vanishes mid-turn with no traceback | an MLX Metal abort under memory pressure, below Python | close memory-heavy apps and re-run; lower `CHAD_CTX_SAFETY` or `--model` a smaller quant. The crash report lands in `~/Library/Logs/DiagnosticReports/` and is worth attaching to a bug |
| Disk full of old model weights | the Hugging Face cache keeps every revision | `hf cache ls` / `hf cache rm` |
| `~/.cache/chad/kv` filling up | one KV checkpoint per project, plus one per cleanly quit session | self-evicts at 8 GB; lower it with `CHAD_KV_CACHE_MAX_GB` |
| `~/.chad/checkpoints` filling up | shadow-git snapshots taken before every edit so `/undo` can revert | swept 30 days after a workspace's last edit; safe to `rm -rf` sooner (you lose `/undo` history) |

## Something you configured isn't happening

Most of these are deliberate: chad does not auto-enable things that run code you did not
write, or that cost context you did not ask to spend.

| You see | What's happening | Reach for |
|---|---|---|
| A project `./.mcp.json` contributes no tools | by design: a just-cloned repo must not auto-launch a subprocess | `/mcp trust` in that directory; `/mcp` shows each server and why it is gated |
| A hosted connector shows `needs login` | OAuth servers never auto-connect; the first login is interactive | `CHAD_MCP_OAUTH=1`, then `/mcp login <server>` |
| An MCP server can't find its API key | a stdio server inherits a minimal env allowlist, not chad's environment | declare the variable in that server's `env:` block |
| A skill doesn't show up as `/<name>` | missing description, unparseable YAML, or skills are off | `/skills` prints the list with discovery warnings; check `CHAD_NO_SKILLS` |
| `CHAD_DISABLE names unregistered lever(s)` at startup | a typo'd lever name is a hard error on purpose | `chad levers` prints the valid names |
| `gh`, a deploy script, or anything needing a token fails inside `bash` | credential-shaped env vars (`…_TOKEN`, `…_SECRET`, `…_API_KEY`) are withheld from bash children; chad lists them at startup | `CHAD_NO_ENV_GUARD=1` for that session |
| `git push` fails with `Permission denied (publickey)` inside `bash` | `SSH_AUTH_SOCK` is withheld | run it yourself as `!git push`, or `CHAD_NO_ENV_GUARD=1` |
| `aws` runs against the wrong account | `AWS_PROFILE` is withheld, so the CLI falls back to `[default]` | pass `--profile`, or `CHAD_NO_ENV_GUARD=1` |
| A legitimate command fails with a permissions error under `--yolo` | yolo bash runs in a macOS Seatbelt sandbox: writes confined to the workspace, temp dirs and caches | `CHAD_NO_SEATBELT=1` if the sandbox itself is the problem; if it is `.git`, check `CHAD_PROTECT_GIT` |
| `pip install chad` installed something else | the PyPI package is **`chad-code`**; bare `chad` is an unrelated package | `uv tool install chad-code` (the command is still `chad`) |
| Wondering what it actually did | the full, redacted trace is off by default | `CHAD_SESSION_LOG=1`, then read `~/.chad/session.log` |

## The through-line

A small local model rewards a scoped ask. "Fix the failing test in `tests/test_x.py`"
lands; "improve my codebase" flails. When in doubt, shrink the task and name the file.
