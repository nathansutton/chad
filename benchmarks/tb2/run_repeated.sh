#!/bin/bash
# N back-to-back TB2 passes (the n>=3 rule: never publish a single sample).
# Each pass is a full run_tb2.sh invocation with its own timestamped .tsv.
#
# Usage:  CHAD_BASE_URL=http://host.docker.internal:8080/v1 \
#           ./run_repeated.sh [N] [MAXCAP] [REPEATS] [TASK]
#   N       number of full passes (default 3)
#   the remaining args are passed straight to run_tb2.sh (see its header)
set -u
cd "$(dirname "$0")"
N="${1:-3}"; shift || true
for pass in $(seq 1 "$N"); do
  echo "=== run_repeated: pass $pass/$N start $(date) ==="
  ./run_tb2.sh "$@"
done
echo "=== run_repeated: all $N passes done $(date) ==="
echo "--- per-pass tallies (newest last) ---"
for f in $(ls -t tb2_full_*.tsv | head -n "$N" | tail -r 2>/dev/null || ls -t tb2_full_*.tsv | head -n "$N"); do
  printf "%s  " "$f"
  awk -F'\t' 'NR>1 && $4!="ERR"{n++; s+=$4} NR>1&&$4=="ERR"{e++} END{printf "pass_rate=%.1f%% (%.1f/%d) errors=%d\n", (n?100*s/n:0), s, n, e+0}' "$f"
done
