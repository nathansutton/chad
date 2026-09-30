# Installing & using chad

How to install and upgrade chad, what the terminal UI does, and the command-line flags.

## Before you start

- An Apple Silicon Mac with 24 GB of RAM or more, on macOS 14 or later.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/), which installs and
  runs chad.
- About 30 GB of free disk for the first run: a ~14 GB download, plus a ~13 GB converted
  copy that makes later starts fast. chad checks before it downloads.
- `git`, which `/undo` uses to snapshot your files. On a new Mac, `xcode-select
  --install` provides it.

Then, from inside a project:

    chad "fix the failing test in tests/test_parser.py"

A small local model rewards a scoped ask: name the file, and ask for one thing.

## Installing & upgrading

The one-line quickstart is `uvx chad-code`. The other ways in:

```bash
uv tool install chad-code   # install for good, then it's just `chad`
uvx --from git+https://github.com/nathansutton/chad chad   # bleeding-edge main, no clone
```

Every example in these docs is written as `chad …`. That is the command after
`uv tool install chad-code`. With `uvx`, write `uvx chad-code …`; in a clone,
`uv run chad …`. Environment variables go in front either way:
`CHAD_TEMP=0.7 uvx chad-code`.

Or from a clone (the dev path):

```bash
uv sync                      # install deps + the `chad` entrypoint (one time)
uv run chad                  # the terminal UI
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

`chad` opens a terminal UI (built on prompt_toolkit). It pins
an input box and a status row at the bottom and prints into your terminal's normal
scrollback, so scrolling and copying work as usual:

- shift-tab cycles permission modes: `normal` (confirm each bash/write/edit) →
  `auto-accept edits` (edits land silently, **terminal commands still ask**) → `yolo`
  (nothing asks) → `plan mode` (read-only: investigate and propose a numbered plan) → back.
  A finished plan lands in `./plans/`; ctrl-g (or `/accept`) clears the context and starts
  implementing it ([details](configuration.md#plan-mode)).
- Steering. Text you type while a turn runs is given to that turn after its current step,
  so you can redirect it without stopping it. A `!command` typed mid-turn waits and runs
  after.
- Approving takes a deliberate key. `y` and `n` answer a pending approval only when the
  input box is empty; **v** prints the whole command or diff first, and **esc** denies at
  any time. Text you were typing can still be sent with Enter.
- ctrl-c interrupts the running turn without killing the session; on an empty prompt,
  press it twice to quit. ctrl-d and `/exit` quit at once, and if a turn is running
  they stop it and save the conversation first. **↑prefilled / ↓generated** token
  counts show an advancing **%** on an unavoidable full re-prefill, so it is never
  silent.
- `@file` / `@dir` mentions and `!command` shell passthrough. Pull a file into
  context inline, or run a shell command without invoking the model.
- Standing project instructions. A `CLAUDE.md` (or `AGENTS.md`) in the working directory is
  appended to the system prompt, and `/init` reads the project and writes one for you
  ([details](configuration.md#project-instructions-claudemd--agentsmd)).
- Voice mode, all local. `/speech`, then ctrl-t to talk: Parakeet-on-MLX transcribes into
  the input box for you to review before Enter sends it, and replies are read aloud via macOS
  `say`. A word table teaches it your identifiers. Needs the `speech` extra
  ([details](configuration.md#voice-mode-speech)).

| Key | What it does |
|---|---|
| Enter | send |
| alt-enter, ctrl-j | new line in the message |
| shift-tab | cycle the permission mode |
| esc, ctrl-c | interrupt the running turn |
| ctrl-c, with no turn running | clear the input; on an empty prompt, press twice to quit |
| ctrl-g | accept a pending plan |
| ctrl-t | start and stop dictation (with `/speech` on) |
| ctrl-d | quit (a running turn is stopped and saved first) |

`chad --help` is the source of truth:

| Flag | What it does |
|---|---|
| `-c, --continue` | resume this directory's **most recent** session (non-destructive) |
| `--resume` | list recent sessions, pick one by number (interactive TTY only) |
| `--plan` | start in read-only plan mode (investigate and propose, edits blocked) |
| `--yolo` | auto-approve bash/write/edit (skip confirm prompts) |
| `--no-think` | skip the model's `<think>` blocks, faster on well-scoped work |
| `--think-budget N` | soft-cap each step's `<think>` at N tokens, force-close it and carry on (off by default) |
| `--backend llama` | run the same harness against a remote llama.cpp server, with `--base-url`, `--tokenizer` and `--api-key-env` ([details](configuration.md#alternate-backend-remote)) |
| `--model` | `auto` (the shipped default), a HF repo id, a local model dir, or a GGUF (a `.gguf` path or `owner/repo/file.gguf`) |
| `--repl` | plain line REPL instead of the TUI |

Two subcommands, each with its own `--help`: `chad prove` (the offline smoke test) and
`chad levers` (print the result-channel lever registry as JSON, for A/B ablation).

A one-shot task (`chad "task"`) runs once and exits. With a terminal on stdin it asks
before each command or edit, as the TUI does; with no terminal on stdin (CI, a pipe,
`</dev/null`) it approves them itself and says so. Every conversation is persisted under
`~/.chad/sessions/`, and every resume forks a new branch rather than overwriting.

| Exit status | Meaning |
|---|---|
| `0` | the task ended on its own |
| `1` | chad stopped it (a guard fired, or the budget ran out), or startup failed |
| `130` | interrupted |

When stdout is not a terminal, only the final answer is written to it; the model's
reasoning and tool trace go to stderr, uncoloured. `NO_COLOR=1` turns colour off
everywhere.
