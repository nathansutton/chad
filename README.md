# chad: a coding agent for your macbook pro

[![tests](https://github.com/nathansutton/chad/actions/workflows/tests.yml/badge.svg)](https://github.com/nathansutton/chad/actions/workflows/tests.yml)

<img src="https://raw.githubusercontent.com/nathansutton/chad/main/docs/claude-vs-chad.jpg" width="50%" alt="Two staircase newel posts side by side: Claude is a hand-carved wooden horse head, chad is a scuffed plastic toy horse tied on with twine">

> Claude can do anything, for anyone, anywhere. chad does one thing. 🗿
> *Coding under supervision.*

chad is a coding agent that runs entirely on an Apple Silicon Mac via
[MLX](https://github.com/ml-explore/mlx). One 27B model and no API key.
(Not affiliated with Anthropic.)

```bash
uvx chad-code          # runs anywhere; `uv tool install chad-code` makes it `chad`
uvx chad-code prove    # offline smoke test: 4 tiny fix-it tasks, verified, timed 🗿
```

The first run asks, then downloads the model once. On a Mac with 24 GB or more that is the ~14 GB build (it needs ~30 GB free the first time); on a 16 GB MacBook Air or Mac mini chad picks the ~8 GB ternary build of the same model instead (it needs ~11 GB free). The PyPI package is `chad-code`; the 16 GB default and `chad-serve` are this fork's, so install it from here: `uvx --from git+https://github.com/riteshdhemla/chad@mac-16gb chad`. See [chad on a 16 GB Mac](docs/16gb.md).

![chad fixing a failing test end to end: reason, read, edit, run pytest, confirm green, all on a local model](https://raw.githubusercontent.com/nathansutton/chad/main/docs/demo.gif)

> Real session, unedited.

## Why chad

Plenty of harnesses run local models now, and [pi](https://pi.dev) is a fantastic default
for the same reason llama.cpp is: it works with everything. chad is moving the opposite direction.

**1 set of silicon**.  This project is focused on making the macbook pro you already have usable.  Not a $10K GPU.

**1 capable model**. `Qwen 3.8 27B`.  This isn't the frontier, but you probably aren't solving frontier problems.  Focus on 1 model buys _speed_.  You'll experience ~ 50 tokens/second generation in a real session instead of ~ 10 tokens/second for stock llama.cpp implementations.  This speed comes from MLX, a couple of targeted custom kernels for this model, and a bundled dflash2 drafter.  The weights are Unsloth's `UD-Q3_K_XL` GGUF and are read natively into MLX.

**1 tightly-coupled agent loop**.  Instead of a standard /completions endpoint, the agent loop in chad owns the backend process.  This comes with nice advantages that make the KV cache more stable and the coding experience measurably better (no long prefills!).  

## Why not

You do not have an Apple Silicon with 16 GB RAM.  You want to pick your local model.  You need a frontier model in a data center.  The list goes on. 

## Documentation

- [Installing & using chad](https://github.com/nathansutton/chad/blob/main/docs/usage.md) covers install, extras and upgrades, the
  terminal UI, and the command-line flags.
- [Throughput & performance](https://github.com/nathansutton/chad/blob/main/docs/benchmarks.md) has every number above, the stock-engine
  comparison, the model, and how to reproduce them with `chad-bench`.
- [Bonsai 2 27B + DFlash2 on the Pi coding agent](docs/pi.md) is this fork's addition: `chad-serve`
  exposes the engine as an OpenAI-compatible server, with setup steps for Pi on a 16 GB Mac.
- [Design](https://github.com/nathansutton/chad/blob/main/docs/design.md) is the argument: why the agent owns the engine, why there are
  five tools, and what 1.x got wrong.
- [Architecture](https://github.com/nathansutton/chad/blob/main/docs/architecture.md) is the module map, the session file format and the
  tool-call wire format.
- [Configuration reference](https://github.com/nathansutton/chad/blob/main/docs/configuration.md) documents project instructions, Agent
  Skills, MCP servers, plan mode, the slash commands, the context window, every environment
  variable, and the safety opt-outs.
- [Troubleshooting](https://github.com/nathansutton/chad/blob/main/docs/troubleshooting.md) maps symptoms to knobs for when a session
  rambles, loops, or slows.
- [Contributing](https://github.com/nathansutton/chad/blob/main/CONTRIBUTING.md) says what lands easily and what needs a conversation
  first.
