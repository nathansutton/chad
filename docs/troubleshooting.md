# Troubleshooting

chad runs a small model on your laptop, so a session can ramble, loop, or slow down in
ways a frontier API won't — and because 2.0.0 is a lean, shell-first harness, most of what
goes wrong is either **memory**, **scope**, or **something you configured that chad
deliberately did not auto-enable**. Every knob that fixes those already exists; this page
maps what you *see* to the one to reach for. The knobs themselves are documented in the
[Configuration reference](configuration.md); this is the symptom index, not a
re-explanation.

**Start here if something is broadly wrong:** `chad prove` runs four tiny fix-it tasks
against the shipped model, offline, and reports what worked. It separates "my install or my
machine is broken" from "the model struggled with my task" in about a minute.

## The model is slow, or won't stop

| You see | What's happening | Reach for |
|---|---|---|
| Long `<think>` before every action | reasoning model, thinking on by default; measured at **~62–66% of all generated tokens** | `--no-think` for well-scoped work. To trim only the rambles, `CHAD_THINK_CEILING=384` first — it force-closes a runaway block but keeps decoding the action in the *same* step, so nothing is re-derived. `--think-budget 1500` is the blunter version that ends the step (see [Turn budgets & think-cap](configuration.md#turn-budgets--think-cap)) |
| A turn burns minutes without landing an edit | runaway turn | the [governor](configuration.md#turn-budgets--think-cap) already ends it for one-shot runs; interactively, ctrl-c and re-scope the ask smaller |
| The same `sed -n` / `rg` run over and over | the loop guard counts identical calls across the whole turn and will abort it — the ask is likely too vague | a smaller, concrete task; **name the file** you want changed |
| First turn in a **new** project takes over a minute before anything happens | cold warm-prefix: the system+tools KV has to be built once per distinct system prompt | expected once, then it is ~5 s. The banner's `[warm start: N prefix tokens from disk cache]` tells you which turn you are having (see [benchmarks](benchmarks.md#the-second-session-in-a-project-starts-warm)) |
| First token slow after `/compact` or a truncation | one-time full re-prefill (the cache is non-trimmable); the status line shows an advancing % | expected — this is the cache trade chad lives with; see [Design & internals](design.md#trimmable-vs-append-only-the-cache-trade-chad-lives-with) |

## Memory

| You see | What's happening | Reach for |
|---|---|---|
| Mac swap-storming with other apps open | ~12 GB of weights plus a ~4.3 GB prefill transient plus your apps exceed free RAM | close apps; or lower `CHAD_CTX_SAFETY` to buy headroom back from the context window (see [Context window](configuration.md#context-window-agentic-coding-needs-room)) |
| The banner reports far less context than 262k | expected: the governor sizes the window from *live* free memory, and 24 GB lands near ~56k | nothing — the banner states what you actually got. More RAM is the only real lever; `CHAD_CTX_SAFETY` trades headroom for window in the other direction |
| chad vanishes mid-turn — no error, **no traceback at all** | an MLX Metal abort under memory pressure; the crash happens below Python, so nothing can print | close memory-heavy apps and re-run; on a borderline box lower `CHAD_CTX_SAFETY`, or point `--model` at a smaller quant. The crash report (worth attaching to a bug) lands in `~/Library/Logs/DiagnosticReports/` |
| Disk full of old model weights | the Hugging Face cache keeps every revision | `hf cache ls` / `hf cache rm` (older CLIs: `huggingface-cli scan-cache` / `delete-cache`) |
| Disk filling up in `~/.cache/chad/kv` | one warm-prefix KV checkpoint per project, ~51 MB floor each | it self-evicts at 8 GB; lower it with `CHAD_KV_CACHE_MAX_GB`, at the cost of a cold first turn in the least-recently-used projects |

## Something you configured isn't happening

Most of these are deliberate: chad does not auto-enable things that run code you didn't
write, or that would cost context you didn't ask to spend.

| You see | What's happening | Reach for |
|---|---|---|
| A project `./.mcp.json` contributes no tools | **by design** — a just-cloned repo must not be able to auto-launch a subprocess. Project-scope servers stay gated until the project is trusted | `/mcp trust` in that directory. `/mcp` shows each server's status and why it's gated. (User-scope `~/.chad/mcp.json` servers auto-connect — you wrote them) |
| A hosted connector shows `needs login` | OAuth servers never auto-connect, because the first login is interactive and would hang a turn | `CHAD_MCP_OAUTH=1`, then `/mcp login <server>` |
| An MCP server that needs an API key can't find it | a stdio server inherits a **minimal env allowlist**, not chad's environment | declare the var in that server's config `env:` block. `CHAD_MCP_FULL_ENV=1` exists but hands every secret to every server |
| A skill doesn't show up as `/<name>` | discovery skipped it (missing description or unparseable YAML), or skills are off | `/skills` prints the list *with* discovery warnings. Check `CHAD_NO_SKILLS` isn't set |
| chad refuses to start: `CHAD_DISABLE names unregistered lever(s)` | a typo'd lever name — this is a hard error on purpose, since it would otherwise run the unmodified harness and report a delta of zero | `chad levers` prints the eight valid names (it loads no model) |
| `gh`, a deploy script, or anything needing a token fails inside `bash` | credential-shaped env var names (`…_TOKEN`, `…_SECRET`, `…_API_KEY`) are filtered out of bash children | `CHAD_NO_ENV_GUARD=1` for that session |
| A legitimate command fails with a permissions error under `--yolo` | yolo bash runs under a macOS Seatbelt sandbox: writes confined to the workspace, temp dirs and caches; reads and network open | `CHAD_NO_SEATBELT=1` if the sandbox itself is the problem. If it's `.git` specifically, check you haven't set `CHAD_PROTECT_GIT` |
| `pip install chad` installed something else | the PyPI package is **`chad-code`**; bare `chad` is an unrelated squatted package | `uv tool install chad-code` (the command is still `chad`) |
| Wondering what it actually did | the full, redacted trace — **off by default** (privacy-first) | enable with `CHAD_SESSION_LOG=1`, then read `~/.chad/session.log` (rotated) |

## The through-line

A small local model rewards a **scoped** ask. "Fix the failing test in `tests/test_x.py`"
lands; "improve my codebase" flails. This is not a limitation chad papers over — it is why
the harness is shaped the way it is: five tools, a shell the model already knows, and a
context window it is worth being frugal with. When in doubt, shrink the task and name the
file.
