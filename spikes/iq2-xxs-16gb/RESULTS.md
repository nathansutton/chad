# IQ2_XXS on a 16 GB Mac — results

**Verdict: IQ2_XXS doesn't work on this machine** (16 GB **M1**). It loads and fits (no swap growth) but failed the
Phase 2 gate at **2/9** (cpp 2/3, go 0/3, rust 0/3; 7 capped), and on the Phase 1 task it thought for 30 min without a
single tool call. UD-IQ2_S did the smoke task (~26 min, verified) but hit **4 Metal OOMs** in prefill (recovered), so
per the plan it stopped there. The ternary did the smoke task fastest (~20 min, verified, no OOM, decode 15.0 tok/s
in the bench), then went 0/3 on h3 rep 1 before the run was cancelled.

Nothing here supports pointing PR #124's contributor at IQ2_XXS instead of the ternary. On this M1 both
are wall-cap-bound at ~5 tok/s in the agent, and the gate as designed can't tell them apart. A 16 GB **M2** (or
a longer `--wall-cap`) is still needed to answer the original question.

## Machine

| | |
|---|---|
| model | Macmini9,1 (**M1**, not the M2 the plan assumed: ~68 GB/s vs ~100 GB/s memory bandwidth) |
| memory | 17179869184 B (16 GB) |
| OS | macOS 15.6 (24G84) |
| commit | f3c88bb9dfe82bdfb28c72a327e15def65a2c2d1 (`main`) |
| disk free at start | 29 GiB (plan asks for ~35 GB) |

Commands: `sysctl hw.model hw.memsize machdep.cpu.brand_string`, `sw_vers`, `git rev-parse HEAD`, `df -h .`

## Phase 0: setup

- Toolchains from native arm64 Homebrew (`/opt/homebrew/bin/brew install go node openjdk@17 cmake rustup`, `rustup default stable`).
  The machine also has an x86 Homebrew at `/usr/local` and an x86 JDK 19; neither was used. The
  arm64 openjdk@17 was put first on `PATH`, with `JAVA_HOME` pointing at it.
- `uv sync && uv run pytest -q` → `1453 passed, 192 skipped, 2 warnings in 102.13s`
- `uv run python benchmarks/polyglot/prepare.py` → `ready: 225 exercises {'cpp': 26, 'go': 39, 'java': 47, 'javascript': 49, 'python': 34, 'rust': 30}`
- `uv run python benchmarks/polyglot/gold.py` → `gold gate: 215/225 tasks valid (1114 s of test time)`, exit 1;
  the rewritten `manifest.json` is byte-identical to the committed one, so the same 10 known
  exclusions fail and all six toolchains are good.

## Phase 1: smoke test

`export CHAD_MODEL=unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ2_XXS.gguf`

### First load (download + one-time conversion)

`uv run chad --yolo "<trivial task>"` (headless one-shot, since the TUI can't be driven from an agent)

- Banner: `chad needs the model ... (~8 GB ...)`, then a note that the first start saves a converted copy (~7 GB) in `~/.cache/chad`.
  **The plan's disk budget leaves out this copy:** `~/.cache/huggingface` grew to 7.9 GB and `~/.cache/chad` to 6.8 GB.
- `ready in 127.0s | context 262144 tokens (compact at 8192)`, measured after the download finished
  (download ~3 min on this connection). No RAM warning printed.
- **Swap during the conversion:** went from 0.6 GB to a **peak of 8.06 GB** (18:53:28), and free memory fell to 9–10%.
  Confounded: `gold.py` was compiling at the same time, and Chrome, Music and System Settings were open.
  That was my mistake, and it's why every Phase 1 measurement below comes from a clean second load.
- The trivial prompt ("reply READY and do nothing else") was a poor smoke test. chad's guard rejected
  ending without a verified change, so the model began exploring the directory. Killed; not counted.

### Second load + one real task (clean: no other heavy apps, nothing compiling)

```sh
export CHAD_MODEL=unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ2_XXS.gguf CHAD_SESSION_LOG=1
cd <fresh scratch git repo>
uv run chad --yolo "Write a Python module roman.py with functions to_roman(n: int) -> str and from_roman(s: str) -> int for 1..3999, plus pytest tests in test_roman.py, then run the tests with pytest and make sure they pass."
```

- `ready in 6.2s | context 262144 tokens (compact at 8192)`
- **Governor floors at 8192 on 16 GB.** From `~/.chad/session.log`:
  `GOVERNOR ctx_limit=8,192 | budget=11.45 GB active=8.47 GB transient=4.30 GB host_avail=3.77 GB kv=65536 B/tok`.
  That gives 11.45 × 0.975 − 8.47 − 4.30 = −1.6 GB for KV, so the window lands on the 8192 floor.
  Every session compacts at 8k tokens (on 24 GB, Q3_K_XL gets about 74k).
- `MEMORY clamp: wired=11.45 GB, limit=10.54 GB`
- **M1-specific:** `QSDPA self-check FAILED (hkv=2 n=3 mlx.core.float16: err=nan, tol=0.004) — fused quantized-KV kernel is numerically broken on this GPU/toolchain; refusing to install it`.
  chad falls back to stock quantized attention. This affects any model on this machine, not only this quant.
- Drafter: `DFlash drafter: no bundle with these weights; borrowing nathansutton/Qwen3.8-27B-DFlash2-MLX's` (4-bit, block 8).
- Warm start: `CACHE warm-start partial: 2190 prefix tokens (disk KV cache, 10.6s)`.
- **Result: did not finish. Stopped at 30 min with no tool call made and no file written.** The whole
  30 minutes was one think block of ~30 KB (~7.5k tokens) that kept re-deriving Roman numeral
  arithmetic with errors ("So MMMCMCMXCIX = 3099. That's not 3999!") and re-planning without acting
  ("Let me write the plan first. Actually, let me reconsider the test file."). No malformed tool
  calls, because none were attempted. No think-tag leakage into output.
- Swap: 0.54 GB at idle before load, 0.74 GB right after load, then **flat or falling (0.61–0.73 GB) for
  the whole 30 min of generation. No swap growth.** `memory_pressure` free: 83% idle, 19–21% while generating.

### Throughput: `uv run chad-bench` (defaults)

```
config                   kv cache fp16 (65,536 B/tok) | prefill chunk adaptive (base 512)
model load                  6.3 s
1. prefill (cold)          5000 tok in 158.34 s   ->      32 tok/s
2. decode                   128 tok in  12.85 s   ->    10.0 tok/s
3. warm step                 16 new tok prefilled (5127 cached)
                           4.21 s of prefill for the follow-up turn
```

The default bench reports one decode figure and **neither the serial/drafted split nor draft acceptance**,
so those two cells are missing. Swap peaked at 0.82 GB during the bench (no growth).

### Phase 1 table

| metric | IQ2_XXS on M1 16 GB |
|---|---|
| context window (banner) | 262144 effective, **compact at 8192** (governor floor) |
| first load | 127.0 s after a ~3 min download (conversion included) |
| second load | 6.2 s (chad), 6.3 s (chad-bench) |
| RAM warning | none |
| swap, idle → generating | 0.54 → 0.61–0.74 GB, no growth (first-load conversion peaked at 8.06 GB, confounded) |
| memory_pressure free, idle → generating | 83% → 19–21% |
| prefill | 32 tok/s |
| decode | 10.0 tok/s (chad-bench, drafter loaded) |
| draft acceptance | not reported by the default bench |
| real task (roman.py) | **not finished in 30 min; zero tool calls** |

**Draft acceptance (from Phase 2 traces, since the default bench doesn't report it):** decode inside the agent ran
4.2–6.5 tok/s per step (trial totals 5.1–5.8 tok/s). That's about half the 10 tok/s `chad-bench` reports.
Draft acceptance was 28–48% while thinking (1.5–2.5 accepted/round) and 49–76% while acting (2.8–5.3 accepted/round).

## Phase 2: harness-3 × 3

Note: `run.py`'s one-block guard (`server.refuse`, `pgrep -f benchmarks/polyglot/run.py`) excludes only its own
pid and its parent. Wrapping the command as `caffeinate -dims uv run python benchmarks/polyglot/run.py ...`
makes the `caffeinate` process (and the calling shell) match, so the block refuses to start with
`another polyglot block is running (pid [26555]): one at a time`. Worked around without a source change:
run the block from a script file and use `caffeinate -dims -w $$` alongside it.
Labels say `m2` as the plan wrote them; the machine is an M1.

```sh
CHAD_MODEL=unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ2_XXS.gguf \
  uv run python benchmarks/polyglot/run.py --label iq2xxs-m2-h3 \
    --tasks-file benchmarks/polyglot/subsets/harness-3.txt --reps 3
```

| task | rep | passed | capped | wall s | steps | gen tok | peak ctx |
|---|---|---|---|---|---|---|---|
| cpp/grade-school | 1 | ✅ |  | 1058.9 | 14 | 3992 | 8100 |
| go/scale-generator | 1 | ❌ | cap | 1200.0 | 3 | 5820 | 5603 |
| rust/alphametics | 1 | ❌ | cap | 1199.7 | 11 | 4699 | 8177 |
| cpp/grade-school | 2 | ✅ |  | 807.5 | 6 | 3597 | 7953 |
| go/scale-generator | 2 | ❌ | cap | 1199.9 | 2 | 5607 | 5990 |
| rust/alphametics | 2 | ❌ | cap | 1199.8 | 3 | 4300 | 6912 |
| cpp/grade-school | 3 | ❌ | cap | 1200.4 | 8 | 5571 | 7132 |
| go/scale-generator | 3 | ❌ | cap | 1200.0 | 2 | 6913 | 5249 |
| rust/alphametics | 3 | ❌ | cap | 1199.9 | 4 | 4997 | 5123 |

```
$ python3 stats.py score _runs/iq2xxs-m2-h3/trials.jsonl
3 tasks, 9 trials (3.0 reps/task)
pass rate   22.2%   (mean of per-task rates)
  cpp         66.7%   n=1
  go           0.0%   n=1
  rust         0.0%   n=1
always pass 0 | never pass 2 | frontier 1
hit the wall cap: 7 trials | median wall 1200 s | total 2.9 h
```

**Gate: 2/9 → failed** (0–2 means too damaged). For reference, on the same 9-trial block on 24 GB: UD-Q3_K_XL 9/9,
UD-IQ3_XXS 8/9, ternary (`fetch.py --label tern-chad`, `stats.py score`) 4/9 (cpp 3/3, go 1/3, rust 0/3; 5 capped).
Caveat: 7 of the 9 failures ran out the 1200 s wall cap at ~5 tok/s on an M1. On a faster machine the same
quant would get more steps in the same wall time, so part of this gap is the machine. The traces below
show it isn't only speed, though: go/scale-generator stalled in one long think block in all three reps.

Swap during the 2.9 h block went from 0.77 GB to a peak of 1.03 GB (small, no paging). The `thermal` column
was empty in every row. `levers_fired`: `trim_spill` on cpp and rust, `bash_line_clip` 2–8× per rust trial,
nothing on go (it barely acted).

### Failed trials (`python3 trace.py <trajectory>`)

- **go/scale-generator rep 1:** it read the stub and tests once (step 4), then spent 1,069 s in a single
  5,695-token think block at 5.3 tok/s, planning the implementation, until the wall cap cut it off (`interrupted`).
  It never wrote code. The step-3 action shows `<tool_call> <function=b…` inside the think block.
  `total: think 5756 tok, decode 1084 s, prefill 110 s, 5.4 tok/s, cache hit 75%, 0 compactions`.
- **rust/alphametics rep 1:** it spent nine steps paging through one long line of the test file
  in byte-offset slices (`tail -c +1350 /tmp/line113.txt`, `tail -c +500 ... | head -c 300`, ...), compacted once
  when context reached 8177 (step 9: 8177 → 3748), ran `cargo test` against the untouched stub, then went back to grepping.
  It never wrote a solution. Step 13's action renders as raw `<tool_call> <function=bash> <parameter=command> ...`
  text, which looks like a malformed tool call the parser didn't pick up.
  `total: think 430 tok, decode 201 s, prefill 220 s, 5.1 tok/s, cache hit 90%, 1 compactions`.
- For contrast, **cpp/grade-school rep 1 (pass):** `think 2095 tok, decode 628 s, prefill 252 s, 5.8 tok/s, 2 compactions`.
  It used 88% of the wall cap on a task the ternary passed in 600 s on 24 GB.

## Phase 4: UD-IQ2_S (8.37 GB)

Run from a script that does, in order: the Phase 1 roman.py task on a first load (`CHAD_SESSION_LOG=1`, 30 min cap
after `ready`), then `uv run chad-bench`. Same task text as above, `CHAD_MODEL=unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ2_S.gguf`.
Clean machine this time: no other heavy apps, nothing compiling.

### Phase 1 (IQ2_S)

- First load (download + conversion): `ready in 40.9s | context 262144 tokens (compact at 8192)`. **Swap stayed flat
  at 0.74 GB during the conversion**, which points to the 8 GB peak in the IQ2_XXS first load coming from
  `gold.py` compiling in parallel, not from the conversion.
- `GOVERNOR ctx_limit=8,192 | budget=11.45 GB active=9.57 GB transient=4.30 GB host_avail=3.32 GB kv=65536 B/tok`.
  That's 1.1 GB more resident than IQ2_XXS, and the window is on the same 8192 floor. The same QSDPA self-check failure applies, with the same borrowed drafter.
- **Metal OOM during prefill, 4 times, each caught and retried with a smaller chunk; no crash:**
  ```
  22:48:13 WARNING PREFILL Metal OOM at 0/559 (kv=5657): retrying with chunk=235
  22:48:23 WARNING PREFILL Metal OOM at 0/559 (kv=5657): retrying with chunk=117
  22:58:09 WARNING PREFILL Metal OOM at 0/1037 (kv=6452): retrying with chunk=206
  22:58:28 WARNING PREFILL Metal OOM at 206/1037 (kv=6658): retrying with chunk=103
  ```
  The plan lists "hits a Metal OOM" as a stop condition. These were recovered, not fatal, but they show IQ2_S
  is at the edge of the Metal budget on 16 GB once the KV reaches about 5.6k tokens.
- **Real task: finished.** `exit=0`, 1628 s wall from launch (download included) and about 26 min after `ready`, 16 steps.
  Final answer: "Created roman.py ... and test_roman.py; all 25 pytest tests pass, verified by a full 1..3999 round-trip check."
  Verified independently: `uv run python -m pytest -q test_roman.py` → `25 passed`; round-trip over 1..3999 holds;
  `to_roman(3999) == 'MMMCMXCIX'`. Along the way it hit a real bug (888 → `DCCC`), wrote its own round-trip
  check, and fixed the bug. Tool calls were well formed and nothing leaked out of the think block.
- Decode in the agent: 5.5–6.1 tok/s per step; draft acceptance about 0.41–0.42 per turn.
- Swap: 0.735–0.743 GB for the whole run, **no growth**.
- `uv run chad-bench` (IQ2_S):
  ```
  model load                  6.0 s
  1. prefill (cold)          5000 tok in 212.75 s   ->      24 tok/s
  2. decode                   128 tok in   9.89 s   ->    12.9 tok/s
  3. warm step                 16 new tok prefilled (5127 cached)
                             1.57 s of prefill for the follow-up turn
  ```
  Swap 0.735 GB flat during the bench. No OOM during the bench.

| metric | IQ2_XXS | IQ2_S |
|---|---|---|
| resident (`active`) | 8.47 GB | 9.57 GB |
| compact at | 8192 (floor) | 8192 (floor) |
| first / second load | 127.0 s (confounded) / 6.2 s | 40.9 s / 6.0 s |
| swap during generation | flat | flat |
| Metal OOM | none | 4 prefill OOMs, all recovered by chunk retry |
| prefill / decode (chad-bench) | 32 / 10.0 tok/s | 24 / 12.9 tok/s |
| decode in agent | 4.2–6.5 tok/s | 5.5–6.1 tok/s |
| roman.py task | not finished in 30 min, zero tool calls | **finished in ~26 min, 25 tests pass (verified)** |

**Stopped Phase 4 at IQ2_S on the owner's call** (Metal OOMs = stop). IQ2_S's gate block was not run. Afterwards the
owner deleted the IQ2_XXS and IQ2_S caches (`~/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF`,
`~/.cache/chad/gguf/Qwen3.8-27B-UD-IQ2_{XXS,S}-*`; about 30 GB) and asked for the ternary pack to be
tested on this machine with the same protocol.

## Ternary Bonsai 2 pack on the same M1 16 GB

`CHAD_MODEL=nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX`, same script and same roman.py task as Phase 4.

- First load (download): `ready in 123.1s | context 262144 tokens (compact at 8192)`.
- `GOVERNOR ctx_limit=8,192 | budget=11.45 GB active=8.76 GB transient=4.30 GB host_avail=3.03 GB kv=65536 B/tok`,
  the same 8192 floor as both quants.
- It uses its own bundled drafter (`.../Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX/.../dflash`) instead of borrowing one.
  The QSDPA self-check fails the same way.
- **Real task: finished.** `exit=0`, 1532 s wall from launch (download included); `ready` 09:10:00 → `DONE accepted` 09:30:15,
  so **about 20 min, 13 steps**. Final answer: "Wrote roman.py (to_roman/from_roman for 1..3999; from_roman rejects
  non-canonical forms) and test_roman.py (15 pytest tests ...); pytest passed 15/15 and a full-range 1..3999 round-trip
  sanity check passed." Verified independently: `15 passed`; round-trip over 1..3999 holds; `to_roman(3999) == 'MMMCMXCIX'`.
  First tool call came about 2 min after `ready`.
- **No Metal OOM.** Swap 0.62–0.64 GB, flat.
- `uv run chad-bench` (ternary): load 5.6 s; prefill 5000 tok in 186.33 s → **27 tok/s**; decode 128 tok in 8.51 s → **15.0 tok/s**;
  warm step 1.45 s. Swap 0.62 GB flat.

| metric | IQ2_XXS | IQ2_S | ternary |
|---|---|---|---|
| resident (`active`) | 8.47 GB | 9.57 GB | 8.76 GB |
| compact at | 8192 | 8192 | 8192 |
| Metal OOM | none | 4 (recovered) | none |
| prefill / decode (chad-bench) | 32 / 10.0 | 24 / 12.9 | 27 / 15.0 tok/s |
| roman.py | not finished (30 min, 0 tool calls) | finished ~26 min | finished ~20 min, 13 steps |

### Ternary harness-3 (cancelled by the owner after rep 1)

```sh
CHAD_MODEL=nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX \
  uv run python benchmarks/polyglot/run.py --label tern-m1-h3 \
    --tasks-file benchmarks/polyglot/subsets/harness-3.txt --reps 3
```

| task | rep | passed | capped | wall s | steps | gen tok | peak ctx |
|---|---|---|---|---|---|---|---|
| cpp/grade-school | 1 | ❌ | cap | 1199.9 | 4 | 5415 | 6812 |
| go/scale-generator | 1 | ❌ | cap | 1200.0 | 3 | 6034 | 5616 |
| rust/alphametics | 1 | ❌ | cap | 1199.9 | 12 | 4145 | 8119 |

0/3 on rep 1, all capped. IQ2_XXS's rep 1 on the same machine was 1/3. Three trials can't separate the two.
- **cpp/grade-school rep 1:** three bash steps exploring, then a 3,072-token think block (577 s) about the 656 KB
  Catch2 header (`test/catch.hpp`). It never edited the solution. `think 3245 tok, decode 655 s, prefill 84 s, 5.4 tok/s`;
  draft acceptance 38% think / 30% act. In the agent the ternary decodes at about 5.4 tok/s, the same as IQ2_XXS,
  even though `chad-bench` says 15.0.

## What surprised me / what I could not do

- **The machine is an M1, not an M2.** Every quant and the ternary decoded at about 5–6 tok/s inside the agent, and with
  that and an 8k compaction trigger, most trials run out the 1200 s cap. On this hardware the wall cap, more than the
  quant, decides most trials.
- **On 16 GB the governor puts every candidate on the 8192 floor** (8.47–9.57 GB resident + 4.30 GB transient vs an
  11.45 GB Metal budget).
- **The QSDPA fused kernel fails its NaN self-check on M1** for every model, and chad falls back to stock attention.
- **`chad-bench` decode overstates agent decode by 2–3×** (10–15 tok/s bench vs 5–6 in the agent at 28–48% draft
  acceptance), and the default bench reports neither the serial/drafted split nor acceptance.
- The plan's disk budget leaves out the ~7–8 GB converted copy each GGUF quant writes to `~/.cache/chad`.
- `run.py`'s one-block guard fires on an ancestor `caffeinate` whose argv contains the script path (see Phase 2).
- **Not done:** Phase 3 (h36 paired runs), IQ2_S's Phase 2, UD-Q2_K_XL, and the remaining 6 ternary h3 trials.
  So there is **no same-machine paired comparison** between IQ2_XXS and the ternary.
