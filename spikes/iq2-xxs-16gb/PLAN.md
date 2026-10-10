# Does Unsloth's UD-IQ2_XXS make chad usable on a 16 GB Mac?

Handoff plan for an agent working from a fresh clone of this repo on a **16 GB M2 Mac
mini**. Everything it needs is in this file and the repo; there is no other context.

## Why

chad's default model is Unsloth's `Qwen3.8-27B-UD-Q3_K_XL.gguf` (13.2 GB). With its
1.2 GB drafter and a ~2 GB prefill transient, it does not fit a 16 GB Mac.

PR #124 (an outside contribution) proposes making the ternary Bonsai 2 pack
(`nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX`, ~8 GB resident) the default
below 24 GB. chad already shipped that pack as its default in 2.2.0 and reverted it in
2.3.0. On the 36-task polyglot subset it passed 24 where UD-Q3_K_XL passed 34
(p = 0.002), and all twelve of its failures ran out the 20-minute wall clock. See the
2.3.0 entry in `CHANGELOG.md` and the `tern-chad-h36` / `q3kxl-chad-h36` rows in
`benchmarks/polyglot/RUNS.md`.

The question here is whether a small Unsloth quant of the same model does better on
16 GB than the ternary. The candidate is the smallest 2-bit file:

    unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ2_XXS.gguf    7.27 GB

chad decodes IQ2_XXS natively (`src/chad/mlx_gguf.py`, `FORMATS`), but nobody has run
this file in chad yet. **If it holds up, the owner declines PR #124 and points the
contributor at this quant.** Your job is to produce the evidence for that call, not to
make it.

## Ground rules

- **Do not change source code** under `src/`. This is a measurement. If something breaks,
  write it up (Phase 1 says what to record) and stop. Don't patch around it.
- **Git identity.** If you commit anything, commit as
  `Nathan Sutton <8629697+nathansutton@users.noreply.github.com>`. Set it per repo
  (`git config user.name` / `user.email`) before your first commit. Work on the branch
  this file is on (`spike/iq2-xxs-16gb`). Never push to `main`.
- **One engine at a time.** Close other heavy apps. On 16 GB, a browser with many tabs
  changes the result. Don't run two benchmark blocks at once (`run.py` refuses anyway).
- **Record, don't round.** Write every number into `spikes/iq2-xxs-16gb/RESULTS.md` as
  you get it, with the command that produced it.

## Phase 0: set up the Mac (about 30 min, plus downloads)

1. Xcode command line tools: `xcode-select --install`.
2. [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`.
3. Benchmark toolchains (`benchmarks/polyglot/prepare.py` checks for `python3`, `go`,
   `cargo`, `node`, `npm`, `java`, `javac`, `cmake`, `c++`), for example with Homebrew:
   `brew install go node openjdk@17 cmake rustup && rustup-init -y`. Put the JDK on `PATH`
   as brew's caveat says. A missing toolchain makes its language's tasks get *skipped*,
   which breaks pairing with the published runs, so get all six.
4. From the repo root: `uv sync`, then `uv run pytest -q`. It must pass before you go on.
5. Disk: you need about 35 GB free (IQ2_XXS 7.3 GB, ternary ~8 GB, drafter sidecar
   1.2 GB, toolchain caches, the 4 GB benchmark KV cache under
   `benchmarks/polyglot/_kv`).
6. Record the machine in RESULTS.md: `sysctl hw.model hw.memsize machdep.cpu.brand_string`,
   `sw_vers`, and `git rev-parse HEAD`.

## Phase 1: smoke test (about 1 hour). Does it load, fit, and work at all?

Point everything at the candidate with one env var. `CHAD_MODEL` reaches every entry
point (`chad`, `chad-bench`, `benchmarks/polyglot/run.py`) through
`chad.cli._pick_model`, and the polyglot run records it in `meta.json`:

```sh
export CHAD_MODEL=unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ2_XXS.gguf
```

1. **First load.** `uv run chad` in a scratch directory. The first start downloads the
   file plus the sidecar and converts arrays once (slow, minutes). Later starts are
   seconds. Record:
   - the banner's context window (the governor sizes it from free memory; on 24 GB the
     Q3_K_XL lands around 74k)
   - first-load and second-load time
   - whether a RAM warning printed
2. **Memory.** While chad is idle and again while it is generating, record
   `sysctl vm.swapusage` and `memory_pressure | tail -1`. **Any swap growth during
   generation is a red flag.** Note it prominently, because a model that pages its
   own weights is not usable no matter what it scores.
3. **One real task.** In a scratch git repo, give chad a small coding task with tests,
   e.g. "write a Python module `roman.py` with to_roman/from_roman and pytest tests,
   then run the tests". Record whether it finished, the wall time, and whether the
   tool calls were well-formed. The banner/status shows tok/s, and so does the
   end-of-turn line. Watch for garbled tool calls or think tags leaking into output,
   which is the typical way a 2-bit quant degrades.
4. **Throughput.** `uv run chad-bench --help`, then run its default bench with
   `CHAD_MODEL` still set, and record decode tok/s (serial and drafted) and draft
   acceptance rate. For reference, on a 24 GB M4 Pro Q3_K_XL decodes ~12 tok/s serial,
   ~46 drafted. Expect an M2 base mini to be well below that (less memory bandwidth).
   What matters is the ratio against the ternary on *this* machine (Phase 3).

**Stop and report if:** the model fails to load, chad crashes or hits a Metal OOM, swap
grows during generation, or the one real task produces malformed tool calls repeatedly.
Write the failure (full stderr, the command, the memory readings) into RESULTS.md. That
alone answers the question ("no").

## Phase 2: quick gate (about 1–3 hours). harness-3 × 3

```sh
uv run python benchmarks/polyglot/prepare.py     # once
uv run python benchmarks/polyglot/gold.py        # once; proves toolchains, no model
uv run python benchmarks/polyglot/run.py --label iq2xxs-m2-h3 \
    --tasks-file benchmarks/polyglot/subsets/harness-3.txt --reps 3
python benchmarks/polyglot/stats.py score benchmarks/polyglot/_runs/iq2xxs-m2-h3/trials.jsonl
```

References on this same 9-trial block, all on a 24 GB Mac: UD-Q3_K_XL 9/9, UD-IQ3_XXS
8/9, ternary: see the `tern-chad` row in RUNS.md (`fetch.py --label tern-chad`, then
`stats.py score`).

**Gate:** 0–2 of 9 means the quant is too damaged. Record it, skip to Phase 4, and
report. 3 or more of 9 means go on. Use `trace.py` on a failed trial to say *how* it
failed (wrong code, malformed calls, ran out the wall cap).

## Phase 3: the real comparison (overnight, about 2 × 6–12 hours). h36, paired on this Mac

The published h36 runs were on a faster 24 GB Mac, and failures there were wall-cap
timeouts, which depend on the machine. So run **both** the candidate and the ternary here,
on the same 36 tasks, back to back:

```sh
T=benchmarks/polyglot/subsets/harness-36.txt

CHAD_MODEL=unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ2_XXS.gguf \
  uv run python benchmarks/polyglot/run.py --label iq2xxs-m2-h36 --tasks-file $T --reps 1

CHAD_MODEL=nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX \
  uv run python benchmarks/polyglot/run.py --label tern-m2-h36 --tasks-file $T --reps 1

cd benchmarks/polyglot
python stats.py score _runs/iq2xxs-m2-h36/trials.jsonl
python stats.py score _runs/tern-m2-h36/trials.jsonl
python stats.py compare _runs/tern-m2-h36/trials.jsonl _runs/iq2xxs-m2-h36/trials.jsonl
```

Blocks are resumable: rerun the same command after a crash or sleep. Disable sleep for
the duration (`caffeinate -dims` in another terminal). Each row records the machine's
thermal state, so note anything odd.

Record for each arm:
- pass count
- capped count (wall-cap timeouts)
- median wall per trial
- median generated tokens
- median peak context
- the `compare` output (sign test p-value, tasks up / down)

## Phase 4: if IQ2_XXS fails the gate (optional, only if time allows)

Try the next sizes up, one at a time, Phase 1 then Phase 2 only:
`UD-IQ2_S` (8.37 GB, about the ternary's size), then `UD-Q2_K_XL` (9.83 GB). Same
`CHAD_MODEL=unsloth/Qwen3.8-27B-GGUF/<file>` form. Stop at the first that clears Phase 1
without swap and gets ≥ 3/9 in Phase 2, and recommend it for a Phase 3 run.

## Deliverable

`spikes/iq2-xxs-16gb/RESULTS.md`, committed on this branch, with:

1. **Verdict, first line:** one of
   - "IQ2_XXS is at least as good as the ternary on 16 GB"
   - "IQ2_XXS is worse than the ternary"
   - "IQ2_XXS doesn't fit or doesn't work"
   - "IQ2_XXS failed; <other quant> looks viable"
   
   Follow it with the numbers that support it.
2. Machine, commit, and every command run.
3. Phase 1 table: context window, load times, swap and memory pressure, tok/s, draft
   acceptance.
4. Phase 2 and Phase 3 score tables and the compare output.
5. Two or three failed trials described from `trace.py`: what went wrong.
6. Anything that surprised you or that you could not do.

Do **not** publish runs to Hugging Face (`publish.py --upload`) or edit `RUNS.md`. The
owner decides what gets published. Leave `_runs/` in place on the machine. Commit only
RESULTS.md, then push the branch:
`git push origin spike/iq2-xxs-16gb`.
