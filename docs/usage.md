# Installing & using chad

## Before you start

- An Apple Silicon Mac with 16 GB of RAM or more, on macOS 14 or later. 24 GB and up
  runs the full build; a 16 GB Air or mini runs the smaller ternary build, chosen for
  you ([what changes on 16 GB](16gb.md)).
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/), which installs and
  runs chad.
- About 30 GB of free disk for the first run on 24 GB and up: a ~14 GB download plus a
  ~13 GB converted copy that makes later starts fast. On 16 GB it is about 11 GB: one
  ~9 GB download, no converted copy. chad checks before it downloads.
- `git`, which `/undo` uses to snapshot your files (`xcode-select --install` on a new Mac).

Then, from inside a project:

    chad "fix the failing test in tests/test_parser.py"

A small local model rewards a scoped ask: name the file, and ask for one thing.

## Installing & upgrading

```bash
uvx chad-code                                              # run it, nothing installed
uv tool install chad-code                                  # install for good; then it's `chad`
uvx --from git+https://github.com/nathansutton/chad chad   # bleeding-edge main
```

Every example in these docs is written as `chad …`, the command after
`uv tool install chad-code`. With `uvx`, write `uvx chad-code …`; in a clone, `uv run chad …`.
Environment variables go in front either way: `CHAD_TEMP=0.7 uvx chad-code`.

Two extras are opt-in because they pull dependencies not every install wants: `speech`
(voice mode) and `highlight` (syntax colour in diffs). An extra rides on the install spec:

```bash
uv tool install --force 'chad-code[speech]'   # add to an existing install
uvx --from 'chad-code[speech]' chad           # one-off run
uv sync --extra speech                        # from a clone
```

`/speech` in the TUI prints whichever of those matches your install.

Upgrade with `uv tool upgrade chad-code`, `uvx --refresh chad-code`, or `git pull && uv
sync` in a clone. What changed is in [`CHANGELOG.md`](../CHANGELOG.md). Model weights are
versioned separately, so a code upgrade never re-downloads the model.

## The terminal UI

`chad` opens a full-screen TUI that pins an input box and a status row at the bottom and
prints into your terminal's normal scrollback, so scrolling and copying work as usual.

- **Permission modes**, cycled with shift-tab: `normal` (confirm each bash/write/edit) →
  `auto-accept edits` (edits land silently, commands still ask) → `yolo` (nothing asks) →
  `plan mode` (read-only: investigate and write a numbered plan to `./plans/`, then ctrl-g
  or `/accept` implements it; [details](configuration.md#plan-mode)).
- **Steering.** Text typed while a turn runs is given to that turn after its current step.
  A `!command` typed mid-turn runs after the turn.
- **Approving takes a deliberate key.** `y` and `n` answer a pending approval only when
  the input box is empty; `v` prints the whole command or diff first; esc denies.
- **Interrupting.** ctrl-c stops the running turn without killing the session; on an empty
  prompt, twice quits. ctrl-d and `/exit` quit at once, stopping and saving a running turn
  first. An unavoidable full re-prefill shows an advancing % in the status row.
- `@file` / `@dir` pull a file into context inline; `!command` runs a shell command without
  the model.
- A `CLAUDE.md` or `AGENTS.md` in the working directory is appended to the system prompt;
  `/init` writes one ([details](configuration.md#project-instructions-claudemd--agentsmd)).
- **Voice mode**, all local: `/speech`, then ctrl-t to talk. Parakeet-on-MLX transcribes
  into the input box for you to review, and replies are read aloud with macOS `say`. Needs
  the `speech` extra ([details](configuration.md#voice-mode-speech)).

| Key | What it does |
|---|---|
| Enter | send |
| alt-enter, ctrl-j | new line |
| shift-tab | cycle the permission mode |
| esc, ctrl-c | interrupt the running turn |
| ctrl-c, no turn running | clear the input; on an empty prompt, twice quits |
| ctrl-g | accept a pending plan |
| ctrl-t | start and stop dictation (with `/speech` on) |
| ctrl-d | quit (a running turn is stopped and saved first) |

## Command line

`chad --help` is the source of truth.

| Flag | What it does |
|---|---|
| `-c, --continue` | resume this directory's most recent session |
| `--resume` | list recent sessions and pick one by number (needs a TTY) |
| `--plan` | start in read-only plan mode |
| `--yolo` | auto-approve bash/write/edit |
| `--no-think` | skip the model's `<think>` blocks; faster on well-scoped work |
| `--think-budget N` | soft-cap each step's `<think>` at N tokens, force-close it and carry on |
| `--backend llama` | run the same harness against a remote llama.cpp server, with `--base-url`, `--tokenizer` and `--api-key-env` ([details](configuration.md#alternate-backend-remote)) |
| `--model` | `auto` (the shipped default), a HF repo id, a local model dir, or a GGUF (`.gguf` path or `owner/repo/file.gguf`) |
| `--repl` | plain line REPL instead of the TUI |

Two subcommands, each with its own `--help`: `chad prove` (the offline smoke test) and
`chad levers` (print the result-channel lever registry).

A one-shot task (`chad "task"`) runs once and exits. With a terminal on stdin it asks
before each command or edit; with no terminal on stdin (CI, a pipe, `</dev/null`) it
approves them itself and says so. Every conversation is saved under `~/.chad/sessions/`,
every resume forks rather than overwrites, and a session you quit cleanly leaves its KV
cache on disk so resuming it does not re-read the conversation.

| Exit status | Meaning |
|---|---|
| `0` | the task ended on its own |
| `1` | chad stopped it (a guard fired or the budget ran out), or startup failed |
| `130` | interrupted |

When stdout is not a terminal, only the final answer is written to it; the reasoning and
tool trace go to stderr, uncoloured. `NO_COLOR=1` turns colour off everywhere.
