# Design & internals

*The engineering story behind Chad — why it's built the way it is. For install and
usage, see the [README](../README.md). For measured numbers, see
[benchmarks](benchmarks.md).*

## Why Chad exists

The interesting engineering in a local coding agent isn't the tool use — it's making a
small model on a laptop feel *responsive* in an agentic loop. That comes down to one
idea:

**Context is not free, and prefill is the bill.**

Every turn of an agentic loop, the model has to *read* the entire conversation so far
before it can write a single new token. That read is the **prefill**: running the
transformer forward over every token in the prompt to build the **KV cache** (the
per-token attention state the model needs to keep generating). Decoding — actually
emitting text — is memory-bandwidth bound and roughly constant. Prefill is the part
that balloons: every step appends the model's reply, the tool call, and the tool's
output to the transcript, so a naive backend re-reads an ever-longer prompt *every
step*. That is O(n) work per step and O(n²) over a session.

Concretely, on a 24 GB M4 Pro at Ornith's ~500 tok/s prefill: by step 20 a real coding
session is ~5,000 tokens of transcript. Re-prefilling all of it is **~15 seconds of
dead air before the model says anything — every step, and growing.** Over a 40-step
task, prefill (not generation) is where the minutes vanish.

Chad's answer is a **persistent prefix KV cache**: keep the KV state alive across turns
and diff each new prompt against what's already cached, so you only prefill the handful
of *appended* tokens.

```
step N prompt:  [ system + tools | cwd · CLAUDE.md | turn 1 | … | turn N-1 | turn N ]
                └──────────────── already in the KV cache ───────────────┘ └─ new ─┘
                          prefill 0 tokens (reused verbatim)              prefill ~30
```

Same session, ~30 new tokens per step instead of 5,000: **~0.1 s of prefill per step
instead of ~15 s.** That gap is the entire reason a 35B model on a laptop can feel like
an agent instead of a batch job.

### Why prefill is *hard*, not just expensive

The cache only helps if the new prompt is a strict *extension* of the cached one. Two
things fight that, and Chad handles both:

- **Compaction.** Long sessions overflow the context window, so old tool output must be
  trimmed — which changes the prefix and would normally throw the whole cache away.
  Chad compacts oldest-first and reclaims enough in a single pass that it won't
  re-trigger next step (see [Context window](configuration.md#context-window-agentic-coding-needs-room)).
- **A non-trimmable cache.** Ornith is a hybrid SSM/attention model: its recurrent
  layers carry state that *can't be rewound to an arbitrary earlier token*. The cache
  can only grow by append; any divergence forces a full rebuild. Chad leans into that —
  it reuses by extension and keeps a disk-checkpointed copy of the stable system+tools
  prefix, so even a divergence reloads that ~3k-token base instead of re-prefilling it
  from scratch.

Everything below is how that gets built — and the rest of what it takes to make a small
model act like a coding agent.

## Trimmable vs. append-only: the cache trade Chad lives with

The whole prefill story above hinges on reusing the KV cache. There's a second property of
a KV cache that decides *how* you're allowed to reuse it, and it's worth naming because
Chad's model gives one up: **trimmability** — the ability to rewind the cache to an
arbitrary earlier token and keep going from there.

A **pure-attention** transformer is trimmable. Each token's K/V is computed independently
and stored in its own row, so "rewind to token *k*" is just "discard the rows past *k*."
That cheap rewind unlocks two things that matter on a laptop:

- **Prompt-lookup / speculative decoding (PLD).** Propose a draft continuation (an n-gram
  the model is about to re-quote from context), verify the whole run in one batched
  forward, and on a partial reject *roll the cache back* to the last accepted token. That
  rollback **is** a trim. With no trim, every rejected draft costs a full re-prefill — on
  novel-text-heavy generation that re-feed overhead makes the hybrid path measurably
  *slower* than just decoding, so PLD is the wrong trade without trimmability.
- **Partial reuse on divergence.** When a new prompt diverges from the cache mid-stream —
  compaction trimmed an old tool output, or an edit changed something in the middle — a
  trimmable cache keeps the common prefix and re-prefills only from the divergence point.
  Append-only can't: any divergence is a full rebuild.

**Ornith is not trimmable.** It's a hybrid SSM/attention (`qwen3_5`) model, and its
recurrent layers carry a *fixed-size running state that is a function of the entire
sequence so far*. There's no per-token row to drop, so there's nothing to rewind to;
`cache_utils.can_trim_prompt_cache` reports false and `engine._trimmable` stays off. PLD is
gated on that flag and falls back cleanly — it can never speed up the shipped model.

That's not a pure loss — it's the *same* recurrent design that keeps Ornith's KV footprint
flat (a fixed-size SSM state no matter how long the context grows; see the
[Context window](configuration.md#context-window-agentic-coding-needs-room) table). Chad trades
trimmability for a memory profile that fits comfortably in 24 GB. The job, then, is to stay
fast on an **append-only** cache, which Chad does three ways:

1. **Reuse by *extension* only.** The normal agentic loop only ever *appends* — the model's
   reply, the tool call, the tool output — so each new prompt is a strict extension of the
   cached one and hits the cache verbatim. That's the 99% case, and it's free.
2. **Compaction that protects the prefix.** When the window fills, Chad compacts
   oldest-first and reclaims enough in one pass that it won't re-trigger next step, keeping
   recent turns byte-identical so the cache extension still holds.
3. **A disk-checkpointed stable base.** The system+tools prefix (~3k tokens — the part that
   never changes turn to turn) is persisted to disk keyed by its rendered token ids, with
   the recurrent SSM state serialized (a fixed ~51 MB floor). On a cold start *or* a
   divergence that can't be reused in RAM, that base reloads with **zero prefill** instead
   of being rebuilt from scratch.

So where a trimmable model would lean on PLD and partial-prefix repair, Chad leans on
append-only reuse + a warm on-disk base — and gets the responsive agentic loop anyway.
Implementation lives in `engine.py` (`_trimmable`, `warm_prefix`, the prefix diff).

## Why a language server matters — *more* on a small model

A frontier model can often hold a repo's structure in its head: it reads a few files,
infers the call graph, and keeps "which `save()` is the method and which is the free
function" straight while it works. A 35B model on a laptop can't. It has weaker reasoning
*and* a smaller context window where — as the whole prefill argument above shows — every
extra token it reads to orient itself is wall-clock you pay for. So the two things a small
local model is worst at, **navigation** and **context economy**, are exactly the two
things a language server fixes. That makes the LSP layer disproportionately valuable here:
it's not a nicety, it's how a small model navigates a real repo without flailing.

Chad's code intelligence is a **two-phase symbolic stack**, deliberately split by what
each phase is good at:

- **Structure (tree-sitter, `repomap.py`).** Fast and language-agnostic. `repo_map` ranks
  every definition with a built-in PageRank over the file→symbol reference graph;
  `overview` / `find_symbol` / `view_symbol` / `find_refs` read one symbol at a time. This
  is the cheap, always-available skeleton.
- **Precision (solidlsp/pyright, `lsp.py`).** Semantic resolution that name-matching
  *cannot* give: true go-to-definition, and cross-file find-all-references that follow
  imports, respect scoping, and don't confuse a method with an unrelated function of the
  same name. We drive Serena's synchronous LSP core directly — **no MCP server, no Serena
  agent** — spawning pyright on a worker thread (~0.3 s warm), started **lazily** on first
  use and bound to one project root.

Why each of those earns its place specifically on a *small* model:

- **Fewer steps = less prefill = less dead air.** Every grep-and-read the model does to
  answer "where is this defined / who calls this" appends tokens to the transcript, and
  prefill is the bill (see the top of this doc). A `find_refs` that returns the three
  real call sites replaces a flailing sequence of greps and full-file reads, and a
  `rename_symbol` that rewrites every call site in one call replaces a long, error-prone
  grep-and-edit loop — fewer steps, fewer transcripts re-prefilled.
- **Deterministic ground truth the model doesn't have to reason about.** A small model
  genuinely confuses two same-named symbols. The `repo_map`/`overview` output annotates
  each definition with a short **"used by …" hint** pulled from the LSP, so the model can
  tell them apart *at a glance* instead of guessing — disambiguation baked into the map
  rather than left to weak reasoning.
- **All-or-nothing correctness on multi-site edits.** A rename that misses one of eight
  call sites is a broken build, and a small model doing it by grep+edit *will* miss one.
  `rename_symbol` follows scope and renames every site or none.

**Never a hard dependency.** If the language server can't start — no Node, offline,
unsupported language — every caller falls back to the tree-sitter backend. Python is the
proven backend (pyright); the other languages light up automatically once their server is
available. The LSP is a precision *upgrade*, and the harness runs fine without it.

The honest payoff is **reliability, not raw prefill savings**. On a small repo a capable
model can often just `grep` its way to a name-greppable symbol, so the symbolic tools don't
necessarily cut prefill there — what they buy is fewer flailing steps and a rename that's
correct at every call site instead of one that misses the eighth. The win grows with the
repo: once a name is shared across dozens of look-alikes, `grep` drowns and precise
resolution is the only thing that lands.

## Architecture

The code is a standard `src/` package; tests live in `tests/`:

```
src/chad/        importable package (uv installs it as the `chad` console script)
  cli.py         argument parsing + the one entrypoint (chad.cli:main)
  agent.py       agentic loop + guardrails
  engine.py      MLX inference + persistent prefix cache
  tools.py       bash/read/write/edit/glob/grep toolset
  tui.py         full-screen prompt_toolkit UI
  ...            prompt, render, repomap, symbols, lsp, validate, … (modular)
tests/           pytest suites (uv run pytest)
```

```
cli.py ──▶ agent.py (agentic loop + guardrails) ──▶ engine.py (MLX + persistent prefix cache)
                 │                                          │
                 └─ tools.py (bash/read/write/edit/glob/grep)
```

- **engine.py** — loads Ornith once, keeps its KV cache alive across turns, and on every
  turn diffs the new prompt against the cached token ids so it only prefills the appended
  tokens. That's why multi-step tool loops stay snappy: re-rendering the whole transcript   
  each step prefills ~20–50 new tokens while 5000+ are served from cache.  
- **agent.py** — renders the conversation through the model's chat template (with tool
  schemas), streams the turn, parses tool calls, runs them, feeds results back, loops
  until the model stops calling tools.
- **tools.py** — the Claude-Code-style toolset with JSON schemas.

## What it borrows from the masters

Small local models are flaky tool-callers, so the harness leans on hard-won ideas from
existing agents:

**[forge](https://github.com/antoinezambelli/forge) — a reliability layer for self-hosted tool-calling:**
- **Rescue parsing** — accept `<tool_call>` XML, ```json fences, *and* bare JSON
  objects. (Weaker local coders routinely emit fenced JSON instead of the templated XML.)
- **Argument validation + nudge** — missing required args get a corrective message the
  model can retry against, instead of a crash.
- **No-op guard** — an `edit` where `old == new` is rejected with an explanation.
- **Edit recovery cascade** — dogfooding showed ~1 in 6 `edit` calls missed on mechanical
  near-misses (the model emitting literal `\n`/`\t` in `old`, or indentation/trailing-ws
  drift). `tool_edit` now retries exact → escape-normalized → whitespace-flexible, each
  still requiring a *unique* target (never edits on ambiguity), and returns the closest
  line in the file on a true miss so the model self-corrects instead of looping. Guarded
  by `test_edit.py` (13 cases incl. safety: a wrong/ambiguous `old` must not change a byte).
- **Loop guard** — identical tool calls counted across the whole turn (catches
  alternating `read A / read B` cycles); 3rd repeat nudges, persistent looping aborts
  the turn cleanly instead of spinning forever.

**[opencode](https://github.com/anomalyco/opencode) `beast` prompt — making weaker models agentic:**
- **Persistence** — keep going until the request is resolved; don't yield early.
- **Verify by running** — and "when you say you'll call a tool, actually call it."

**[OpenHarness](https://github.com/HKUDS/OpenHarness) — base prompt structure:**
- Lead-with-the-answer tone, read-before-edit, don't over-engineer, prefer dedicated
  tools over `bash`, and an injected **environment section** (OS/shell/cwd).

**[deepagents](https://github.com/langchain-ai/deepagents) — "batteries included":**
- **Planning tool** (`write_todos`) — for any 2+ step task the model lays out a plan and
  marks items `in_progress`/`completed`. The scaffold keeps a small model on-track and
  acting rather than narrating.
- **Workspace snapshot** — the system prompt injects a listing of the project's files
  (git-tracked or globbed) so the model knows it's in a real repo and explores it. This
  is what flipped the agent from "paste a generic rewrite into chat" to "grep → read →
  edit the actual file."
- **Act-via-tools + verify-before-`done`** — a refactor must go read → edit → run tests;
  the `done` tool is rejected if files were changed but nothing was run to verify them.

**[Claude Code teardown](https://gist.github.com/Haseeb-Qureshi/d0dc36844c19d26303ce09b42e7188c1) — the real thing:**
- **Cache-boundary trick** — static behavioral prompt first (reused by the prefix KV
  cache), volatile per-session context (cwd, `CLAUDE.md`/`AGENTS.md`) after. Mirrors
  Claude Code's "re-send the full prompt every turn, rely on caching" design.
- **Anti-hallucination guardrail** — an *internal-only* Claude Code instruction:
  never claim a test passed / task is done when the output shows otherwise.
- **Context compaction** — when the prompt nears the limit, reclaim space oldest-first:
  strip stale `<think>` reasoning from older assistant turns (Ornith's blocks are large
  and rarely matter later), then head/tail-truncate the oldest large tool outputs,
  keeping recent turns verbatim (`--ctx-limit`). This and verified append-only cache
  reuse are what keep long agentic sessions responsive.
