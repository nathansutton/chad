# Serving the engine to other agents (`chad-serve`)

`chad-serve` puts chad's MLX engine behind an OpenAI-compatible
`/v1/chat/completions`, so [Pi](https://pi.dev) (or any agent that speaks OpenAI chat
completions) runs on the same drafted decode and prefix cache chad uses, without chad's
own agent loop.

It picks the model as `chad` does: the Q3_K_XL build with 24 GB or more, the ternary
Bonsai 2 pack on a 16 GB Mac ([16gb.md](16gb.md)), both with the DFlash2 drafter, unless
`--model` or `CHAD_MODEL` says otherwise.

chad's own loop still owns its engine in-process, and that is the path the project
tunes and measures. `chad-serve` is for when you want a different harness on the same
engine.

This page sets up Pi; the server section applies to any client.

## Requirements

- An Apple Silicon Mac with 16 GB of RAM or more, on macOS 14 or later.
- [`uv`](https://docs.astral.sh/uv/) (`brew install uv`).
- Node.js 20 or later, for Pi (`brew install node`).
- Free disk for the model: about 11 GB on a 16 GB Mac, about 30 GB with 24 GB or more.

## 1. Start the server

```bash
uvx --from chad-code chad-serve     # or `chad-serve` after `uv tool install chad-code`
```

The first run downloads the model into `~/.cache/huggingface`; later starts load in a
few seconds. It is ready when it prints a line like:

```
ready in 5.6s | context 262144 tokens (safe limit 41750)
```

Note the **safe limit**: it is what this machine's memory can hold, and it is the number
to give Pi as the context window in the next step. It lands around 40,000 tokens on a
16 GB Mac and around 74,000 on 24 GB.

The server listens on `http://127.0.0.1:8081/v1`. Leave it running in its own terminal.

From a clone of this repo, `uv run chad-serve` does the same.

| Flag | Default | Meaning |
|------|---------|---------|
| `--port` | `8081` | Port to listen on. |
| `--host` | `127.0.0.1` | Address to bind. There is no authentication; keep it on loopback. |
| `--model` | by RAM | An HF repo id, local model dir or GGUF, as `chad --model` takes it. `CHAD_MODEL` is read when the flag is absent. |

The `CHAD_*` engine variables in [configuration.md](configuration.md) apply unchanged,
for example `CHAD_NO_DFLASH=1` to decode without the drafter, or `CHAD_MAX_CONTEXT` to
cap the window.

## 2. Install Pi

```bash
npm install -g @earendil-works/pi-coding-agent
```

## 3. Point Pi at the server

Create `~/.pi/agent/models.json` (merge the `chad` entry into `providers` if the
file already exists):

```json
{
  "providers": {
    "chad": {
      "baseUrl": "http://127.0.0.1:8081/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "models": [
        {
          "id": "chad",
          "name": "chad (local Qwen3.8-27B)",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 40000,
          "maxTokens": 8192,
          "compat": { "thinkingFormat": "qwen" },
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    }
  }
}
```

Set `contextWindow` a little under the safe limit the server printed. Pi compacts the
conversation as it approaches that number; a prompt over the server's limit is refused
with a `context length exceeded` error rather than risking an out-of-memory crash.

`thinkingFormat: "qwen"` is what lets Pi switch the model's reasoning off: Pi then sends
`enable_thinking` with each request, which the server honours.

## 4. Run Pi

From the project you want to work in:

```bash
pi --model chad/chad
```

Or pick "chad (local Qwen3.8-27B)" under `/model` inside Pi. A one-shot run is
`pi -p --model chad/chad "your task"`.

Add `--thinking off` for routine work. The model then answers without a reasoning block,
which roughly halves the tokens it writes; on a 16 GB Air a module-and-tests task took
55 seconds that way against 159 with thinking on. Leave thinking on for hard problems.

## Check it works

```bash
curl -s http://127.0.0.1:8081/health
curl -s http://127.0.0.1:8081/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":200}' \
  | python3 -m json.tool
```

The response carries a `timings` object (`predicted_per_second`, `cache_n`,
`draft_accept_rate`), and the server logs one line per request:

```
prompt 46 new + 2271 cached (0.7s) | generated 85 at 19.8 tok/s | draft accept 45%
```

`cached` is the part of the prompt served from the prefix cache. In an agent session it
should cover almost everything after the first call; if it stays at 0, the client is
changing earlier turns between requests.

## What to expect

On a base M5 MacBook Air with 16 GB, a five-call bug fix takes about a minute: 9 to 10
seconds to read Pi's first 1,600-token prompt, about a second before the first token on
each later step, and 13 to 30 tokens/second decoding depending on what is being written.
[16gb.md](16gb.md) has the full table, and the reason long tasks slow down on a fanless
machine.

## How it behaves

- **One request at a time.** There is one model and one KV cache; concurrent requests
  queue. The prefix cache follows the most recent conversation, so alternating between
  two sessions re-reads each prompt in turn.
- **Reasoning.** The model thinks unless the request says `enable_thinking: false`
  (top-level, or inside `chat_template_kwargs`). Reasoning is streamed as
  `reasoning_content`. A request's `reasoning_effort` maps onto the template's three:
  `minimal` and `low` run as `low`, `medium` as `medium`, `high` and above as `xhigh`;
  with none sent, the loaded model's default applies. Changing the level
  mid-session changes the system block and costs one full re-read of the prompt.
- **Streaming with thinking off.** A turn that is only a tool call sends nothing until
  the call is complete, because tool calls are returned whole.
- **Sampling.** The model card's recipe for the mode is the default: temperature 1.0,
  top-p 0.95, top-k 20 when thinking, and 0.7 / 0.80 / 20 when not. A request's `temperature`, `top_p`, `top_k`, `min_p` and
  `presence_penalty` override it for that request.
- **Tool calls.** The model writes calls in the Qwen XML dialect; the server returns
  them as OpenAI `tool_calls`, converting non-string parameters by the tool's JSON
  schema.
- **Text only.** Image parts are replaced with a placeholder.

## Troubleshooting

- **`context length exceeded`**: lower `contextWindow` in `models.json` so Pi compacts
  sooner, or free memory (quit browsers) and restart the server to get a higher limit.
- **Slow, with the Mac swapping**: the model needs about 8 GB of the machine's memory.
  Close other heavy apps, and do not run a second local model alongside it.
- **Pi `-p` hangs in a script**: it is waiting on stdin; run it with `< /dev/null`.
- **Port in use**: `lsof -nP -iTCP:8081` shows what holds it; `--port` picks another
  (change `baseUrl` to match).
