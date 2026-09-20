#!/usr/bin/env python3
"""Search latency: ripgrep vs. Microsoft tgrep, on the corpora chad actually runs in.

tgrep is a trigram-indexed grep with a ripgrep-compatible CLI. Its published win is on
100k+ file monorepos; the question here is what it is worth at the corpus sizes chad's
own session history shows, which are three orders of magnitude smaller.

Three tgrep modes are separated because they behave like three different tools:

    no-index  the index is missing or coverage-gated, so tgrep scans like grep
    index     an on-disk index, no daemon
    server    `tgrep serve` warm, queried over TCP

Arms are interleaved per query so thermal drift lands on all of them equally, and each
query is best-of-N to report the floor rather than a scheduler artifact. Match counts are
compared against ripgrep on every query, because a search that is faster and wrong is not
faster.

Run:
    tgrep index <corpus> --index-path /tmp/tgrep-idx/<name>
    uv run python benchmarks/tgrep/latency.py <corpus> --index-path /tmp/tgrep-idx/<name>
"""

import argparse
import json
import os
import statistics
import subprocess
import time

# The real query mix, taken from the rg invocations chad's model emitted across its
# recorded sessions: short literal identifiers, `-n` on almost all of them. Inventing
# regexes here would measure a workload chad does not have.
QUERIES = (
    "timeout", "timeoutMessage", "TimeoutError", "KyOptionsRegistry", "totalTimeout",
    "findUnknownOptions", "InternalOptions", "kyOptionKeys", "throwProcessedError",
    "export", "import", "retry", "known", "Options", "compaction", "run_turn",
    "levers", "ambient", "prefill", "engine",
)
ARMS = ("rg", "noindex", "index", "server")


def argv_for(arm: str, pattern: str, index_path: str) -> list[str]:
    """The command line for one arm. `--` before the pattern is not optional: tgrep
    reads a bare `search`/`index`/`serve`/`status` as a subcommand, and `tgrep -n serve
    file` starts a server and never returns."""
    if arm == "rg":
        return ["rg", "-n", "--", pattern, "."]
    base = ["tgrep", "-n", "--index-path", index_path]
    if arm == "noindex":
        base.append("--no-index")
    return base + ["--", pattern, "."]


def timed(argv: list[str], cwd: str, reps: int) -> tuple[float, int, str]:
    """(best wall-clock ms, stdout line count, error text). stdin is /dev/null because a
    search with no path argument reads it and blocks forever behind a pipe."""
    best = None
    lines = 0
    for _ in range(reps):
        t0 = time.perf_counter()
        try:
            p = subprocess.run(argv, cwd=cwd, capture_output=True,
                               stdin=subprocess.DEVNULL, timeout=120)
        except subprocess.TimeoutExpired:
            return 0.0, 0, "timed out"
        dt = (time.perf_counter() - t0) * 1000
        if p.returncode == 2:
            return 0.0, 0, p.stderr.decode(errors="replace").strip()[:160]
        best = dt if best is None else min(best, dt)
        lines = len(p.stdout.splitlines())
    return best or 0.0, lines, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--index-path", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    corpus = os.path.abspath(os.path.expanduser(args.corpus))
    arms = [a for a in args.arms.split(",") if a in ARMS]
    nfiles = len(subprocess.run(["rg", "--files"], cwd=corpus, capture_output=True,
                                stdin=subprocess.DEVNULL).stdout.splitlines())
    print(f"corpus {corpus}  ({nfiles} files visible to rg)  "
          f"{len(QUERIES)} queries x best-of-{args.reps}\n")

    samples: dict[str, list[float]] = {a: [] for a in arms}
    rg_lines: dict[str, int] = {}
    mismatched = []
    for pattern in QUERIES:
        for arm in arms:
            ms, lines, err = timed(argv_for(arm, pattern, args.index_path),
                                   corpus, args.reps)
            if err:
                print(f"  ! {arm} failed on {pattern!r}: {err}")
                continue
            samples[arm].append(ms)
            if arm == "rg":
                rg_lines[pattern] = lines
            elif pattern in rg_lines and lines != rg_lines[pattern]:
                mismatched.append((pattern, arm, rg_lines[pattern], lines))

    print(f"{'arm':10} {'p50 ms':>9} {'mean ms':>9} {'max ms':>9} {'vs rg':>9}")
    base = statistics.median(samples["rg"]) if samples.get("rg") else 0.0
    p50s = {}
    for arm in arms:
        got = samples[arm]
        if not got:
            continue
        p50 = statistics.median(got)
        p50s[arm] = round(p50, 2)
        rel = f"{base / p50:8.2f}x" if base and p50 else "        -"
        print(f"{arm:10} {p50:9.1f} {statistics.mean(got):9.1f} {max(got):9.1f} {rel}")

    if mismatched:
        print(f"\n{len(mismatched)} match-count mismatches vs rg:")
        for pattern, arm, want, got in mismatched[:10]:
            print(f"  {pattern!r:24} {arm:8} rg={want:6} tgrep={got:6}")
    else:
        print("\nmatch counts identical to rg on every query")

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "_runs",
                                   f"latency-{os.path.basename(corpus)}.json")
    with open(out, "w") as fh:
        json.dump({"corpus": corpus, "files": nfiles, "reps": args.reps,
                   "queries": len(QUERIES), "p50_ms": p50s,
                   "mismatches": len(mismatched)}, fh, indent=2)
    print(f"\nwrote {out}")


main()
