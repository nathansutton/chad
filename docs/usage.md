# Installing & using chad

How to install and upgrade chad, what the terminal UI does, and the command-line flags.

## Installing & upgrading

The one-line quickstart is `uvx chad-code`. The other ways in:

```bash
uv tool install chad-code   # install for good, then it's just `chad`
uvx --from git+https://github.com/nathansutton/chad chad   # bleeding-edge main, no clone
```

Or from a clone (the dev path):

```bash
uv sync                      # install deps + the `chad` entrypoint (one time)
uv run chad                  # full-screen TUI
uv run chad "add a --json flag to main.py and update the tests"   # one-shot, headless
uv run chad -c               # resume this directory's last conversation
```

Two features are opt-in because they pull deps not every install wants:
`speech` (voice mode: a mic library, no torch) and `highlight` (syntax colour in diffs and
previews). An extra rides on the install spec, not on a separate command, so how you add it
depends on how you installed chad:

```bash
uv tool install --force 'chad-code[speech]'   # add to an existing `uv tool` install
uvx --from 'chad-code[speech]' chad           # one-off run, nothing installed
uv sync --extra speech                        # from a clone
```

`/speech` in the TUI prints whichever of those matches your install, so you never have to
work it out from here.

Upgrading depends on how you installed: `uv tool upgrade chad-code`, `uvx --refresh
chad-code`, or `git pull && uv sync` for a clone. What changed lands in
[`CHANGELOG.md`](../CHANGELOG.md). Model weights are versioned separately, so a code upgrade
never re-downloads the model.

For development, `uv sync` once, then `uv run pytest -q`. The fast unit gate loads **no model
weights**, runs in seconds, and is what CI runs. For throughput on your own machine, use
`uv run chad-bench` (see [Throughput & performance](benchmarks.md)).

## Interactive UX

`uv run chad` launches a full-screen terminal UI (built on prompt_toolkit):

- shift-tab cycles permission modes: `normal` (confirm each bash/write/edit) →
  `auto-accept edits` (edits land silently, **terminal commands still ask**) → `yolo`
  (nothing asks) → `plan mode` (read-only: investigate and propose a numbered plan) → back.
  A finished plan lands in `./plans/`; ctrl-g (or `/accept`) clears the context and starts
  implementing it ([details](configuration.md#plan-mode)).
- Type-ahead message queue. Keep typing while the agent works; messages run in order.
- ctrl-c interrupts the running turn without killing the session. **↑prefilled /
  ↓generated** token counts show an advancing **%** on an unavoidable full re-prefill, so it
  is never silent.
- `@file` / `@dir` mentions and `!command` shell passthrough. Pull a file into
  context inline, or run a shell command without invoking the model.
- Standing project instructions. A `CLAUDE.md` (or `AGENTS.md`) in the working directory is
  appended to the system prompt, and `/init` reads the project and writes one for you
  ([details](configuration.md#project-instructions-claudemd--agentsmd)).
- Voice mode, all local. `/speech`, then ctrl-t to talk: Parakeet-on-MLX transcribes into
  the input box for you to review before Enter sends it, and replies are read aloud via macOS
  `say`. A word table teaches it your identifiers. Needs the `speech` extra
  ([details](configuration.md#voice-mode-speech)).

`uv run chad --help` is the source of truth:

| Flag | What it does |
|---|---|
| `-c, --continue` | resume this directory's **most recent** session (non-destructive) |
| `--resume` | list recent sessions, pick one by number (interactive TTY only) |
| `--plan` | start in read-only plan mode (investigate and propose, edits blocked) |
| `--yolo` | auto-approve bash/write/edit (skip confirm prompts) |
| `--no-think` | skip the model's `<think>` blocks, faster on well-scoped work |
| `--think-budget N` | soft-cap each step's `<think>` at N tokens, force-close it and carry on (off by default) |
| `--backend llama` | run the same harness against a remote llama.cpp server, with `--base-url`, `--tokenizer` and `--api-key-env` ([details](configuration.md#alternate-backend-remote)) |
| `--model` | `auto` (the shipped default), or any HF repo id / local model dir |
| `--repl` | plain line REPL instead of the TUI |

Two subcommands, each with its own `--help`: `chad prove` (the offline smoke test) and
`chad levers` (print the result-channel lever registry as JSON, for A/B ablation).

A headless task (positional, or piped with no TTY) auto-approves mutating tools and runs
greedy (temp 0). Every conversation is persisted under `~/.chad/sessions/`, and every resume
forks a new branch rather than overwriting.
