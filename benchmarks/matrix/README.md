# `benchmarks/matrix/` — nine coding harnesses vs. one laptop

Same weights, same MacBook, same eight tasks, same sampler: what does each coding agent
make a local model *read*, and what does that cost the person waiting? This directory is
the whole instrument — the runner, the forcing proxy, the scorecard, the tasks — and the
one run it produced (`_runs/`), so every number in the write-up can be traced to a row.

The question it answers is not "which harness passes more tasks" (the model is the same
in every arm, so that is mostly the weights talking) but **what the harness costs on a
local engine**: how many tokens it prepends to the first turn, how many the server's
prefix cache cannot serve on every later turn, and how many seconds each of those is.

## The run that is committed

Qwen3.8-27B `UD-Q3_K_XL` on an M4 Pro with 24 GB, one `llama-server` shared by every
harness, 8 Exercism tasks, 1,200 s cap, one rep, every harness in its own auto-approve
mode. 88 of 88 cells ran; 19.7 h of wall clock. From `_runs/scorecard.md`:

| Arm                 | tax: turn-1 prompt (tok) | wait before 1st token, turn 1 | uncached tok / later turn (med) | wait / later turn (med · p90) | cache reuse | prefill s / task (med) | exp. tok/s | pass (gate) |
|---------------------|--------------------------|-------------------------------|---------------------------------|-------------------------------|-------------|------------------------|------------|-------------|
| pi+llama            | 2,008                    | 23.3 s                        | 77                              | 1.7 s · 23 s                  | 99%         | 47                     | 8.0        | 7/8 T3      |
| opencode+llama      | 18,057                   | 237.6 s                       | 113                             | 2.5 s · 44 s                  | 99%         | 400                    | 5.9        | 3/8 T5      |
| chad+llama          | 2,562                    | 28.2 s                        | 46                              | 1.0 s · 14 s                  | 99%         | 54                     | 6.9        | 7/8 T1      |
| dsh+llama           | 8,052                    | 96.6 s                        | 100                             | 2.0 s · 34 s                  | 99%         | 141                    | 6.6        | 6/8         |
| goose+llama         | 9,576                    | 119.1 s                       | 3,009                           | 39.4 s · 88 s                 | 76%         | 347                    | 5.0        | 6/8 T4      |
| mini+llama          | 1,171                    | 12.8 s                        | 204                             | 3.0 s · 19 s                  | 96%         | 56                     | 7.9        | 4/8 T4      |
| crush+llama         | 16,263                   | 207.7 s                       | 96                              | 1.9 s · 41 s                  | 100%        | 304                    | 5.5        | 5/8 T3      |
| cline+llama         | 5,876                    | 66.9 s                        | 400                             | 5.6 s · 56 s                  | 96%         | 144                    | 7.1        | 6/8         |
| codex+llama         | 7,804                    | 90.2 s                        | 852                             | 11.6 s · 28 s                 | 91%         | 121                    | 6.3        | 6/8 T2      |
| chad+mlx *          | 2,562                    | 1.2 s †                       | 49                              | 1.1 s · 20 s                  | 99%         | 25                     | 15.6       | 8/8 T1      |
| chad+mlx-nodflash * | 2,566                    | 1.2 s †                       | 35                              | 1.1 s · 14 s                  | 99%         | 25                     | 11.1       | 6/8 T2      |

- **tax** is the first agent request: system prompt + tool schemas + task, in tokens. It is
  paid at least once, and it is how much of a 32k window is gone before you have typed.
- **wait before 1st token** is llama-server's own `prompt_ms` for that request. Every 1,000
  tokens the cache does not hold is ~11 s on this box (prefill 85–95 tok/s, decode ~10,
  flat across every arm).
- **uncached tok / later turn** and **wait / later turn** are the same two numbers on
  agent turns 2+: the part of each turn the cache could not serve, and the pause between
  a tool result landing and the model starting to think.
- **cache reuse** is `cache_n / (cache_n + prompt_n)` on those turns.
- **prefill s / task** counts every request of the task, side requests included.
- **exp. tok/s** is generated tokens over wall clock. n=1 and a ±12% noise floor: nothing
  between pi, chad, mini, cline and codex on this column is a finding. The 10–40x gaps in
  the wait columns are.
- **pass** is a gate, not a ranking, and `T` counts timeouts.
- `*` the two MLX rows are chad on its own in-process engine, self-reported from its
  prefill trace with the same definitions — there is no server to observe. `†` their 1.2 s
  turn-1 wait is a system-prompt prefix restored from disk; the chad+llama row is the cold
  prefill of the same prompt.

The second table in `_runs/scorecard.md` has the shape of each harness (tool count,
system-prompt size, prefix churn, side requests fired beside the agent loop and whether
they overlapped an agent turn), `_runs/tables.md` has pass/wall/prefill/generated totals
and the per-task grid, and `_runs/scorecard.json` has every derived number.

### Things the data says that the first draft of this scorecard got wrong

The first scorecard had two broken columns and this file exists partly so the fixes are
on the record:

1. **The proxy's time-to-first-byte is not the wait.** llama-server emits a first streamed
   chunk before a long prefill finishes, so the proxy's stamp sat at ~30 s for every
   request over ~3k tokens; on 113 of 379 requests it was earlier than the server's own
   `prompt_ms`. The scorecard prints that count and uses `prompt_ms` everywhere.
2. **Side requests polluted the per-turn columns.** opencode, goose and crush fire a
   no-tools title/summary call beside the agent loop, and for opencode and crush it is
   the *first* request of the session. Treating it as turn 1 gave opencode a 657-token
   "first prompt" (real: 18,057) and, worse, made its 18k cold prefill count as a "later
   turn" — which is where an earlier draft's "opencode waits 34 s per turn, 240 s p90"
   came from. With side requests classified out, opencode's later turns are 2.5 s median
   and 99% cached. Its cost is the 238 s of tax, the title call running concurrently
   with it, and the window it leaves behind (exits at 22.5k of 32k). The same fix moves
   crush from "4.6 s · 208 s" to "1.9 s · 41 s". goose is the arm whose cache genuinely
   dies every turn: `cache_n` pins at the system prompt (~9.4k) while the uncached count
   climbs 742 → 1,459 → 5,521 → 8,872 across one task.
3. An earlier note claimed a side request makes the *next* agent turn re-prefill
   thousands of tokens (dsh 2,443 vs 89). Every side request in this run sits beside
   turn 1, and turn 2 is every harness's read of the test file, so that comparison was
   measuring the test-file read. The scorecard now only compares turns 3+, finds no
   sample, and prints nothing. dsh's 8 side requests were abandoned by the client before
   the server answered (counted, not measured).

## Setup, exactly

| | |
|---|---|
| machine | Apple M4 Pro, 24 GB unified memory, macOS 26.6.2, nothing else running |
| weights | `unsloth/Qwen3.8-27B-GGUF` · `Qwen3.8-27B-UD-Q3_K_XL.gguf` (llama arms); chad's MLX conversion of the same recipe (MLX arms) |
| engine | llama.cpp build 10470 (Homebrew), `llama-server -c 32768 -ngl 999 --jinja --metrics`, default 4 slots on a unified KV pool |
| sampler | temp 1.0 · top_k 20 · top_p 0.95 · min_p 0.05 · penalties off — **forced on every request by `sampler_proxy.py`**, audited: one parameter set across all 476 requests (`_runs/sampler_audit_summary.json`) |
| tasks | `tasks/` — 8 Exercism Python exercises, stub + tests + instructions, pristine from git |
| prompt | one sentence, identical for every arm (`PROMPT` in `run.py`) |
| cap | 1,200 s per (arm, task); the process *group* is killed at the cap |
| pass | `pytest -q` on the task's own test file after the harness exits |
| reps | 1 |

Harness versions as run (`_runs/provenance.json`), and how each is installed:

| arm | version | install |
|---|---|---|
| pi | 0.80.3 | `npm i -g @earendil-works/pi-coding-agent` |
| opencode | 1.17.12 | `npm i -g opencode-ai` |
| chad | 2.0.2 | this checkout, `uv sync` |
| dsh (deepseek-harness) | 0.1.1-rc.2 | `npm i -g @deepseek-ai/dsh` |
| goose | 1.39.0 | Block's installer |
| mini-swe-agent | 2.4.6 | `uv tool install mini-swe-agent` |
| crush | 0.92.0 | `npm i -g @charmland/crush` |
| cline | 3.0.60 | `npm i -g cline` |
| codex | 0.151.0 | `npm i -g @openai/codex` |

Three more were installed and dropped at smoke, with the reason in
`_runs/smoke_verdict.json`: **deepagents-code 0.1.65** and **qwen-code 0.22.3** get
`400 failed to parse grammar` from llama-server (their tool JSON schemas, converted to
GBNF under `--jinja`, exceed the rule cap — they cannot drive llama.cpp tool-calling on
this build); **aider 0.86.2** is an edit-block loop with no tool calls and never returns
a server `timings` object, so the instrument cannot measure it.

## Reproduce it

Budget a night: the llama phase alone took 16 h here. One engine resident at a time — a
24 GB box holds exactly one 27B.

```bash
brew install llama.cpp                 # llama-server on PATH
uv sync                                # chad + its venv (every arm gets this venv's python/pytest)
# install whichever harnesses you want as arms (table above); a missing binary is
# skipped by name in provenance, never a crash
export MATRIX_RUNS=benchmarks/matrix/_runs/mine   # keep the committed run intact

uv run python benchmarks/matrix/run.py setup            # write each harness's provider config
                                                        # (your originals kept as *.pre-matrix.bak)
uv run python benchmarks/matrix/run.py smoke            # one short task per llama arm -> smoke_verdict.json
uv run python benchmarks/matrix/run.py llama --from-smoke   # 8 tasks x the arms smoke cleared, ONE server
uv run python benchmarks/matrix/run.py mlx              # chad in-process; llama-server must be down
uv run python benchmarks/matrix/run.py table            # pass / wall / prefill / generated
uv run python benchmarks/matrix/scorecard.py --runs $MATRIX_RUNS   # the felt table + JSON
```

`overnight.sh` is the unattended version of the same sequence (smoke → llama → MLX one
task at a time with a settle window, because load/teardown cycling a 12 GB model has
panicked a GPU here) and refuses to start if a previous run's accumulators are present or
a port is already bound. Run it under `caffeinate -is`.

The GGUF is fetched into the shared Hugging Face cache on first use; `STOCK_GGUF=<path>`
points at one you already have. Setting `MATRIX_MINI_YAML` overrides where
mini-swe-agent's bundled config is looked up (it is otherwise resolved through the
`mini` executable's own interpreter).

### What is and is not committed

Committed under `_runs/`: `grid.json` (one row per cell), `turns.jsonl` (one row per
generation request, from the proxy), `sampler_audit.jsonl` and its summary, `smoke.json`
+ `smoke_verdict.json`, `provenance.json`, `traces/*/prefill_trace.jsonl` (the MLX arms'
in-process equivalent of a turn record), and the rendered `scorecard.md` / `.json` /
`tables.md`. `tests/test_matrix_bench.py` regenerates the two markdown files from the
rows and fails if they differ, so the tables cannot drift from the data by hand.

Not committed: server and proxy logs, the MLX arms' full trajectories (model output),
kept workdirs, captured request bodies, generated provider configs, and superseded runs.
Nothing under this directory names the machine it ran on — paths are stored relative to
the repo, harness output is redacted of the home and temp directories and the hostname,
and the same test scans every tracked file for those.

The `harness` digests in `provenance.json` are of the scripts *as they ran*. The
committed scripts differ from them in three ways, none of which touches what the proxy
forces on or records for a request: the task corpus is vendored here instead of read
from a private checkout, paths and harness output are sanitised before being written,
and `scorecard.py` derives the columns as described above.

## Adding an arm, and diagnosing a cache miss

An arm is one entry in `ARMS` in `run.py` (its documented headless argv), plus, where
needed, one in `ARM_ENV` and one in `ARM_SETUP` (the provider-config writer). The
runner hands every arm the same three files, the same prompt and the same timeout, and
the proxy measures it without its cooperation.

The proxy records the *shape* of each request (system-message and tool-list hashes,
sizes, `cache_n`/`prompt_n`), which names a cache miss caused by a changed system prompt
or tool list but not one caused by the conversation body being re-serialised. For that,
capture the bodies and diff consecutive turns:

```bash
uv run python benchmarks/matrix/run.py smoke --arms goose+llama --capture-bodies
uv run python benchmarks/matrix/body_diff.py --all $MATRIX_RUNS/bodies/goose+llama/grade-school--1/
```

`body_diff.py` reports, per consecutive pair, whether the system message and tools are
byte-identical, the first message index at which the two requests differ, and how many
characters of the rendered conversation they share.

## Caveats, all of them

- n=1 rep, 8 tasks. Throughput noise floor ±12%. The per-turn wait gaps are 10–40x.
- The two MLX rows are self-reported and in-process; their turn-1 number is a
  disk-restored prefix, not a cold prefill.
- The server ran 4 slots on a unified KV pool. The side-request findings are for that
  configuration; a single-slot server would queue them instead, which is a different
  failure, not the absence of one.
- mini-swe-agent does not stream (`stream: false` on every request): whatever the prefill
  number says, the user sees nothing until a turn is done.
- chad+llama drives llama.cpp's raw `/completion` endpoint with token ids, so its
  system/tool hashes are blank in `turns.jsonl`; its 99% reuse and 46 uncached tokens per
  turn are the evidence, and a sceptic cannot diff its prompts from the proxy log.
- `?` on a total in `tables.md` means the server was still busy when that arm's counters
  were read; the number is a floor.
- The GGUF and the MLX checkpoint are the same quantisation recipe, not the same bytes.
- Every harness ran in the auto-approve mode its authors ship. Their prompt and tool
  choices were made for hosted models where prefill is close to free; none of this is a
  bug report against any of them.

Thanks to the llama.cpp maintainers for a server-side prefix cache and a `timings`
object good enough to build the whole instrument on.
