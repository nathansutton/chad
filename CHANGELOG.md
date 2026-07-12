# Changelog

Notable, user-visible changes. Started 2026-07; earlier history is summarized coarsely.

## [Unreleased]

- **The shipped 35B quant is better at the same speed** (plan 075 quant sweep).
  `UD-Q2_K_XL` on Hugging Face now carries 3-bit expert down-projections
  (2-bit gate/up, 6-bit backbone, 8-bit routers; 3.09 bits/weight, ~13.4 GB
  resident, was ~12 GB). Held-out code perplexity drops 7.25 → 6.86 and prefill/
  decode are unchanged (~728 tok/s / ~67 tok/s @8k on a 24 GB M4 Pro) — decode
  reads only the 8 active experts, so the extra bits are effectively free. On the
  TB2 canary set the old build went 2/6, the new one 15/17 model-attributable
  over n=3. The AWQ calibration pass was dropped: measured as neutral-to-negative
  on held-out code (RTN 7.19 vs AWQ 7.25). Existing installs pick the new build
  up on their next model download; local caches keep working.
- **New: a 6-bit build for big-memory Macs** —
  [`Ornith-1.0-35B-Q6-MLX`](https://huggingface.co/nathansutton/Ornith-1.0-35B-Q6-MLX)
  (~28.5 GB; needs ≥48 GB). Built by `benchmarks/tb2/quantize_q6.py`; the TB2
  kit gained a rental-host arm (`serve_q6.sh`, `run_repeated.sh`, README
  section) that serves it on e.g. an EC2 `mac-m4pro.metal`.
- **`chad --serve` — an OpenAI-compatible endpoint on the in-process engine**
  (`/v1/chat/completions`, streaming + non-streaming, `/v1/models`, `/health`).
  Unlike `mlx_lm.server`, it keeps chad's persistent prefix KV cache warm across
  an agent's requests: an append-only follow-up step prefills only the newly
  added tokens (measured on the 35B at a 7.7k-token transcript: 1.1 s warm vs
  11.7 s cold). One request at a time by design; this is the Terminal-Bench 2.0
  Mac arm's server (`benchmarks/tb2/README.md`).
- **35B-on-24GB stability pack (plan 075)**: the engine now installs Metal
  allocator clamps at load (`set_wired_limit`/`set_memory_limit`; opt out with
  `CHAD_NO_MEMORY_CLAMP=1`); a Metal OOM inside a prefill chunk (catchable on
  mlx ≥ 0.32, now required) rolls the cache back exactly, halves the chunk and
  retries instead of killing the process; a mid-decode OOM keeps the partial
  turn and rebuilds cleanly. The prefill chunk size is now adaptive per chunk
  (MoE 2048 / dense 512 base — worth ~+14% prefill on the 35B — decaying toward
  256 as resident context and memory pressure grow; explicit
  `CHAD_PREFILL_CHUNK` still wins). The RAM-aware compaction trigger now also
  respects host-physical free memory (Docker/harbor pressure Metal can't see)
  and re-checks per turn (>10% hysteresis). Compaction that CANNOT get under
  `ctx_limit` (protected floor bigger than the window) now latches instead of
  destroying the warm cache every step — the old behavior cost a measured
  26–28 s full re-prefill per step. Known trade: mlx 0.32.0 itself decodes
  ~2.5% slower than 0.31.2 on the shipped models; taken for the catchable-OOM
  recovery, and more than paid back by the chunk-size default on the 35B.
- **Fixed: temp>0 sampling was frozen on `chad --serve`'s worker thread.** MLX's
  implicit-key RNG never advances on non-main threads (present in stock mlx
  0.32.0), so the server's single engine worker "sampled" with one frozen noise
  vector: identical prompts produced byte-identical responses, and decoding was
  quasi-greedy and markedly loop-prone (per-thread `mx.random.seed()` cannot fix
  it — the seeded state is never consumed there). The engine now samples through
  an explicit split-per-draw key chain (`_KeyedSampler`, fresh entropy per
  generation) whenever temperature > 0, on every thread. Greedy (temp 0) decoding
  is unchanged.
- **Bounded rewind for the hybrid cache (plan 075)**: the engine now snapshots the
  recurrent (DeltaNet) state once per turn at prefill-end — a free reference copy —
  and can rewind to it: restore the recurrent state, native-trim the attention KV,
  re-feed the few agreed-on tokens. A prompt that diverges *inside the last turn*
  (a truncated generation re-rendered, a dropped `--serve` stream, an identical
  prompt retried) now costs at most one turn's re-feed instead of a full-transcript
  re-prefill (tens of seconds at TB2 context sizes). Divergences before the last
  turn (compaction) fall back to the warm-prefix rebuild exactly as before.
  Bit-exactness pinned by `test_hybrid_rewind_matches_fresh` on bf16 hybrid weights.
- **`chad prove` — a two-minute offline smoke test**: four tiny fix-it tasks from
  chad's own dev suite (disclosed as such — it's a proof of life, not a
  benchmark), run through the real agent loop on the validated 9B, mechanically
  verified with tamper-proof re-seeded checkers, and timed. Scorecard reports
  time-to-first-token, decode tok/s, and per-task wall; network access is
  blocked at the library level during task execution; a share snippet is offered
  only on a full pass. Exit codes: 0 all pass / 1 any fail / 2 preflight stop.
  The task set survived a 3-condition falsification gate (3/3 reps under 60 s,
  including memory-constrained) before this shipped.
- **Disk preflight before the model download**: chad now checks free space at the
  Hugging Face cache before offering the 5–12 GB download and refuses with the
  shortfall and the `hf cache rm` pointer instead of dying at 70%; a mid-download
  disk-full failure is now diagnosed as such. The confirm prompt also gained a
  while-you-wait tip (cd into a project, think of a scoped first ask).
- **Unknown-RAM machines get the safe model**: if RAM can't be detected, chad now
  picks the 9B instead of silently falling through to the 35B's 12 GB download.
- **Guard stops now say what to do next**: the loop, repetition, and step-cap
  aborts carry a recovery tip and point at the troubleshooting symptom map
  instead of ending the turn with only a diagnosis.
- **README restructured**: the one-line quickstart now sits at the top, the
  thesis prose after it, and all other install paths under a new
  "Installing & upgrading" section; new
  [Why there's no model picker](docs/design.md#why-theres-no-model-picker)
  section in the design docs.
- **Troubleshooting**: new symptom row for the silent mid-turn exit (an MLX
  Metal abort under memory pressure prints nothing by nature — the row says
  which knob helps and where the crash report lands).
- **Community & releasing**: GitHub Discussions enabled (Q&A + Show and tell),
  issue templates point questions there; `RELEASING.md` documents the release
  checklist; the README demo GIF is regenerated from a checked-in vhs tape
  (`docs/demo.tape`) so the demo can't rot silently.
- **Progress-aware step cap**: a turn that keeps landing *and verifying* changes is no
  longer killed dead at 40 tool steps mid-task — it earns +20-step extensions (absolute
  ceiling 4×) on the warm cache, matching the governor's never-interrupt-real-progress
  rule. A turn that does stop at the cap now banks a progress note (same contract as a
  governor hard-stop), so `continue` / `--auto-continue` resume it instead of the task
  silently dying at the prompt.
- **Responsiveness**: every tool result is bounded before it hits the transcript
  (whole-file reads, wide greps, and the symbolic map tools were each able to stall the
  next turn 20–30s on prefill), and a step that makes several tool calls now shares one
  per-step output budget so the results can't stack into a single giant prefill.
- **Degenerate-repetition stop**: greedy decode locking into repeating one short string
  (minutes of dead generation per occurrence, the main task-timeout cause in dogfooding)
  is now detected within a few hundred tokens; the step is cut off and the model nudged
  out of the loop. `CHAD_NO_REPEAT_GUARD=1` disables (A/B knob).

## [0.1.0] — 2026-07

First coherent public cut. Highlights since `v0.0.1`:

- **Full-screen TUI**: shift-tab permission modes (normal / auto-accept / plan), a
  type-ahead message queue, mid-turn interrupt, and a live status line (↑prefilled /
  ↓generated token counts, ✂N adaptive think-cap counter).
- **Sessions**: resume (`-c`) and list-and-fork (`--resume`, `/resume`) — copy-on-resume,
  so a resume never overwrites the session it branched from.
- **Extensions**: Agent Skills ([agentskills.io](https://agentskills.io)) and MCP servers
  (stdio & Streamable HTTP; OAuth behind `CHAD_MCP_OAUTH=1`), with a project-trust gate —
  a project's `./.mcp.json` servers don't start until you `/mcp trust` them.
- **Engine**: persistent prefix KV cache with warm-prefix disk checkpoints, RAM-aware
  context sizing, an adaptive think-cap (`--think-budget`), and a runaway-turn governor.
- **Safety**: catastrophic-bash seatbelt, deterministic post-edit syntax gate, and a
  minimal environment for stdio MCP subprocesses.
- **Fixed**: `transformers` capped `<5.13` to fix an mlx-lm import crash on fresh installs.
