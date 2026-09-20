# `benchmarks/tgrep/` — why chad still shells out to ripgrep

[tgrep](https://github.com/microsoft/tgrep) is a trigram-indexed grep with a
ripgrep-compatible CLI and an optional server, and it is very good at what it claims: on
a 388k-file checkout of gecko-dev its README reports 33,402 ms → 643 ms against ripgrep.
chad teaches `rg` as the model's first move, so the swap looked worth a spike.

It was measured and it is not worth landing. The reason is not that tgrep is slow — it is
6.6x faster than ripgrep on the largest tree on this machine, and that reproduces. The
reason is that **search is 0.0073% of chad's wall clock**, so the entire prize is four
seconds, and collecting it costs a daemon, a third binary, and four ways to silently
return the wrong answer.

This directory is the instrument and the rows, kept for the same reason
`benchmarks/search/` is kept: so the next person who has this idea can check the
arithmetic instead of rerunning the night.

    workload.py   what chad's own timed session log says search costs      (the denominator)
    latency.py    rg vs tgrep on the corpora chad actually runs in         (the numerator)
    fidelity.py   would a transparent rg -> tgrep rewrite return the same answer?
    _runs/        the output of all three, on an M4 Pro / 24 GB, tgrep 1.0.8, ripgrep 15.2.0

## The denominator kills it before the numerator matters

From `~/.chad/session.log`, which already times every tool call chad makes — 622 recorded
sessions, `_runs/workload.txt`:

| | wall clock | calls |
|---|---|---|
| model decode | 48,043.9 s | 2,594 steps |
| all tool execution | 6,968.2 s | |
| &nbsp;&nbsp;`bash` | 6,965.5 s | 941 |
| &nbsp;&nbsp;**search-only `bash`** | **4.0 s** | **145** |
| &nbsp;&nbsp;grep inside a compound command | 3,244.1 s | 218 |

A call counts as search only when every pipeline segment is a read-only text tool.
That distinction is the whole measurement: `rg -n x src/ && npm test` is a test run
wearing a grep's clothes, and charging its 15 s to search is how you talk yourself into a
7.8% number that is really `npm test`. Search-only latency is p50 20 ms, p90 40 ms,
max 370 ms.

So search is 4.0 s of 55,012 s. **A grep that returned instantly would save 0.0073% of
chad's wall clock**, and the 3x that was actually measured saves 2.7 s — across the whole
recorded history, not per session. Nothing downstream can rescue a lever that starts
here.

## The numerator, for the record

20 queries drawn from the `rg` patterns chad's model really emitted, best-of-3,
arms interleaved (`_runs/latency-*.json`):

| corpus | files | rg p50 | tgrep, no index | tgrep, on-disk index | tgrep, server |
|---|---|---|---|---|---|
| `repo/ky` | 97 | 7.9 ms | 17.1 ms (0.46x) | 4.2 ms (1.89x) | — |
| `repo/chad` | 409 | 17.0 ms | 42.6 ms (0.40x) | 5.7 ms (2.98x) | 6.7 ms (2.48x) |
| `~/assist` | 17,761 | 372.8 ms | — | 56.5 ms (6.60x) | — |

Match counts were identical to ripgrep on every query in every row. Three things in that
table matter more than the speedups:

- **Without an index tgrep is 2.2–2.5x slower than ripgrep.** The index is not an
  optimization, it is a precondition.
- **The server is slower than the bare on-disk index** at chad's scale (6.7 ms vs
  5.7 ms): a TCP round trip costs more than mmap'ing a 5 MB index. The daemon is not
  bought for speed here — see below for what it *is* bought for.
- The 6.6x only appears at 17,761 files. chad's recorded sessions run in
  **570 distinct working directories over 622 sessions — 1.09 sessions per directory**,
  mostly 97–415 files, many of them one-shot temp trees. An index amortizes over reuse
  and there is no reuse to amortize over.

## Four ways the swap returns a wrong answer

chad teaches `rg` and nothing else, because routing follows the model's trained prior and
steering text does not move it (`docs/design.md`). So the only integration shape worth
testing is the transparent one: rewrite `rg …` to `tgrep …` inside the `bash` tool and
leave the prompt alone. `fidelity.py` replays every single-segment `rg` command the model
actually emitted, in the repo its session ran in, under both binaries
(`_runs/fidelity.txt`):

    byte-identical    147     reordered  36     match set differs  2     errored  0
    unparseable        98   (shell quoting shlex cannot split)

98.9% of replayed commands return the same match set. The failures are the interesting
part, and each is silent — no warning, no error, exit code 0:

1. **A stale index answers with nothing.** With an on-disk index and no daemon, a grep
   for a symbol the agent wrote one moment earlier returns no match; ripgrep finds it.
   chad edits constantly, so the daemon is mandatory for *correctness*, not speed —
   which puts a resident watcher process (42 MB RSS after one 20-query run, growing with
   query volume) on a box whose memory is already the binding constraint.
2. **The 64 MiB default cap drops matches.** tgrep skips files above 64 MiB; ripgrep has
   no limit. In the probe, ripgrep finds the needle in both files and tgrep reports only
   the small one, exit 0 either way. `--no-max-filesize` restores parity but must then
   match across `index`, `serve` and every search.
3. **Negative globs anchor differently.** `rg -n TimeoutError source/ -g '!source/errors'`
   excludes that directory; tgrep returns its four files anyway. Spelled `-g '!errors/**'`
   it agrees. `-g` is in the model's real flag mix.
4. **A pattern spelled like a subcommand is parsed as one.** No `--` appears in `rg`
   command lines a model writes, and tgrep reads a bare first positional as a possible
   subcommand:

   | pattern | rg | tgrep | tgrep with `--` |
   |---|---|---|---|
   | `search` | 1 | 4 | 1 |
   | `index` | 2 | 0 | 2 |
   | `status` | 18 | 6 | 18 |
   | `help` | 1 | 0 | 1 |
   | `serve` | 1 | **never returns — starts a server** | 1 |

   `rg -n index …` runs `tgrep index` and rebuilds an index; `rg -n serve …` hangs until
   the tool timeout burns the step. A shim fixes this by inserting `--` before the
   pattern, which requires parsing ripgrep's argument grammar out of a raw shell string —
   and 98 of 283 real commands could not even be split by `shlex`.

## Verdict

Not adopted, and no eval was run to decide it.

That last part is deliberate. The obvious next step is an A/B on the eval rig, and it
would be theater: the rig seeds fresh temp repos of a few dozen files, arms run at k=2,
and a night of wall clock buys a detectable swing of tens of points. The effect under
test is 0.0073% of wall clock and zero of task outcome — thousands of times below the
noise floor of any contrast the rig can draw. A saturated instrument cannot authorize an
adoption, and it cannot authorize a rejection either; the Phase-0 arithmetic decides this
one, which is why it was measured first.

What would change the answer: chad routinely working in trees of 100k+ files, where
ripgrep's per-query cost stops being 20 ms; or tgrep growing a mode that is correct on an
unindexed tree and is not slower than ripgrep there. Neither is true today.

`rg` stays what the prompt teaches, and the ambient hints in `src/chad/ambient.py` keep
tracking `rg`/`grep`/`ag` only.
