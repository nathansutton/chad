# chad: a local Claude-Code-style coding agent for your laptop

[![tests](https://github.com/nathansutton/chad/actions/workflows/tests.yml/badge.svg)](https://github.com/nathansutton/chad/actions/workflows/tests.yml)

<img src="docs/claude-vs-chad.jpg" width="66%" alt="Two staircase newel posts side by side: Claude is a hand-carved wooden horse head, chad is a scuffed plastic toy horse tied on with twine">

> Claude can do anything, for anyone, anywhere. chad does one thing. 🗿
> *Coding under supervision.*

chad is a Claude-Code-style coding agent that runs entirely on an Apple Silicon Mac via
[MLX](https://github.com/ml-explore/mlx). One 27B model, no API key, and nothing leaves the
machine. (Not affiliated with Anthropic.)

```bash
uvx chad-code          # runs anywhere; the command is still `chad`
uvx chad-code prove    # offline smoke test: 4 tiny fix-it tasks, verified, timed 🗿
```

The first run asks, then downloads the model once (~8 GB). chad targets a 24 GB Mac and
nothing smaller. The PyPI package is `chad-code`; bare `chad` is an unrelated squatted
package.

![chad fixing a failing test end to end: reason, read, edit, run pytest, confirm green, all on a local model](docs/demo.gif)

> Real session, unedited (the silent prefill is cut). Recorded with `--yolo`; the default
> mode stops and asks before every edit and every command.

## Why chad

Plenty of harnesses run local models now, and [pi](https://pi.dev) is a fantastic default
for the same reason llama.cpp is: it works with everything. chad goes the opposite way. One
model and one set of silicon, taken to the max, with an inference loop it owns instead of a
server it talks to. What that buys:

- **It is fast enough to talk to.** About 63 tok/s on easy code and about 30 in the middle
  of a real session, where stock llama.cpp gets 11 from the same model on the same Mac. A
  200-token function body is 3 seconds instead of 18
  ([the comparison](docs/benchmarks.md#same-model-same-mac-stock-engine)).
- **A step costs half a second, not most of a minute.** The KV cache persists across
  turns, so each agent step prefills the ~16 tokens it appended instead of the 5,000 it
  already read: ~0.55 s instead of ~48 s. The system prefix is checkpointed to disk, so the
  next session starts warm too ([why prefill is the bill](docs/design.md#why-chad-exists)).
- **It rewinds a cache that cannot be rewound.** This model is a hybrid: most of its layers
  are recurrent, and their state has no per-token rows to drop, so its cache is not
  trimmable. Taken at face value that is an append-only cache, where a rejected draft or a
  diverged prompt means reading everything again. chad holds the model in one process,
  snapshots the recurrent state, trims the attention layers natively, and rolls back
  exactly. That is what lets speculative decoding pay off here, and why a truncated turn or
  a retry costs a few tokens instead of the whole transcript
  ([the cache trade](docs/design.md#trimmable-vs-append-only-the-cache-trade-chad-lives-with)).
- **The window is big enough for real work.** The shipped ternary weights sit at about 8 GB
  resident, which leaves roughly 150k tokens of context on a 24 GB Mac
  ([the model](docs/benchmarks.md#the-model-qwen38-27b)).
- **Your Claude Code habits carry over.** shift-tab permission modes, plan mode,
  `CLAUDE.md` / `AGENTS.md`, `@file` mentions, `!command`, `-c` to resume,
  [Agent Skills](https://agentskills.io) as slash commands and
  [MCP servers](https://modelcontextprotocol.io) from `.mcp.json`
  ([usage](docs/usage.md), [configuration](docs/configuration.md)).

## Why not

- You do not have an Apple Silicon Mac with 24 GB. It runs below that and says so, but a
  16 GB Mac gets a window too small to work in.
- You want to pick your model. `--model` runs any MLX weights, but the drafter and the
  kernels are fitted to the one checkpoint chad ships, so everything else runs slower. Use
  pi.
- You need a frontier model. This is a 27B with five tools (`bash`, `edit`, `write`,
  `write_todos`, `done`). It does what you said, not what you meant: *"fix the failing test
  in `tests/test_x.py`"* lands, *"improve my codebase"* flails.

## Documentation

- [Installing & using chad](docs/usage.md) covers install, extras and upgrades, the
  terminal UI, and the command-line flags.
- [Throughput & performance](docs/benchmarks.md) has every number above, the stock-engine
  comparison, the model, and how to reproduce them with `chad-bench`.
- [Design & internals](docs/design.md) covers why prefill is the bill, the persistent
  prefix cache, and why the tool surface is five tools.
- [Configuration reference](docs/configuration.md) documents project instructions, Agent
  Skills, MCP servers, plan mode, the slash commands, the context window, every environment
  variable, and the safety opt-outs.
- [Troubleshooting](docs/troubleshooting.md) maps symptoms to knobs for when a session
  rambles, loops, or slows.
- [Contributing](CONTRIBUTING.md) says what lands easily and what needs a conversation
  first.
