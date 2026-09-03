#!/usr/bin/env bash
# The harness x engine grid, unattended. One engine resident at a time, because a 24 GB
# box holds exactly one.
#
#   smoke  first — every llama arm on the quickest task, ~2-5 min each. An arm that cannot
#          reach the server, whose turns the proxy cannot measure, or that never touches
#          the stub is dropped HERE with its reason in smoke_verdict.json, instead of
#          spending an hour of the night and then being reported as a loss.
#   llama  second — 8 tasks x the arms smoke cleared, through ONE llama-server, so every
#          arm shares a server-side prefix cache and a single set of weights. It is the
#          long pole and it banks a row after every single run, so being cut short costs
#          the tail tasks and nothing already measured.
#   mlx    last — one invocation PER TASK with a settle window between them. Same rows,
#          same file; `_grid` appends. Split up because this arm loads and tears down
#          ~12 GB of weights each time, and load/teardown cycling on this box has
#          panicked the GPU before.
#
#   caffeinate -is benchmarks/matrix/overnight.sh 2>&1 | tee _runs/overnight.log
set -u
cd "$(dirname "$0")/../.."
# MATRIX_RUNS (run.py honours it too) keeps a new night out of the committed run.
R="${MATRIX_RUNS:-benchmarks/matrix/_runs}"
export MATRIX_RUNS="$R"
mkdir -p "$R"
TIMEOUT=1200
TASKS="bowling,grade-school,affine-cipher,transpose,wordy,book-store,dominoes,go-counting"
# Every arm run.py knows. Whatever is not installed is skipped by name in provenance;
# whatever smoke drops is named in smoke_verdict.json.
ARMS="${ARMS:-$(uv run python -c 'import sys; sys.path.insert(0,"benchmarks/matrix"); import run; print(",".join(run.ARMS))')}"

echo "=== preflight $(date '+%F %H:%M:%S') ==="
echo "free disk: $(df -g /System/Volumes/Data | tail -1 | awk '{print $4}') GB"
echo "wired    : $(vm_stat | awk '/wired/{printf "%.1f GB", $4*16384/1073741824}')"
echo "arms     : $ARMS"
# The accumulators: grid.json rows, the proxy's sampler audit and turn records, the
# MLX arms' traces. Each appends by design, so a previous night's files would fold into
# this one — an audit carrying two samplers fails a clean run, and turn records from
# two nights under one arm name would average two different harness versions.
for f in grid.json smoke.json sampler_audit.jsonl turns.jsonl smoke_verdict.json; do
  if [ -e "$R/$f" ]; then
    echo "REFUSING: $R/$f exists — move the previous run aside (e.g. $R/<date>/) first."
    exit 1
  fi
done
if [ -d "$R/traces" ] && [ -n "$(ls -A "$R/traces" 2>/dev/null)" ]; then
  echo "REFUSING: $R/traces/ is not empty — move it aside with grid.json."
  exit 1
fi
rm -f "$R/sampler_audit_summary.json"
# Warn, never kill: force-sweeping a resident engine is how this box panicked before.
# 8081 is llama-server now and 8080 is the sampler proxy; either one already bound
# means we would silently attach to somebody else's engine.
for port in 8080 8081; do
  if lsof -nP -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1; then
    echo "REFUSING: port $port is already bound:"
    lsof -nP -iTCP:$port -sTCP:LISTEN
    exit 1
  fi
done
if pgrep -fl "llama-server|mlx_lm" | grep -v overnight; then
  echo "WARNING: something may already hold the GPU (above). Stop it by hand and rerun."
  exit 1
fi

step() {
  echo
  echo "=== $1 :: $(date '+%H:%M:%S') ==="
  shift
  local t0=$SECONDS
  if "$@"; then echo "--- ok in $(( (SECONDS-t0)/60 )) min"
  else echo "--- FAILED after $(( (SECONDS-t0)/60 )) min — continuing"; fi
}

wait_server_down() {
  for _ in $(seq 30); do pgrep -f llama-server >/dev/null || break; sleep 2; done
  if pgrep -f llama-server >/dev/null; then
    echo "=== llama-server did not exit; stopping here. ==="
    exit 1
  fi
}

step "setup" uv run python benchmarks/matrix/run.py setup --arms "$ARMS"

step "smoke (one task per llama arm)" \
  uv run python benchmarks/matrix/run.py smoke --arms "$ARMS" --timeout 600 --keep
wait_server_down
echo "--- smoke verdict:"
uv run python -c 'import json, os; v=json.load(open(os.path.join(os.environ["MATRIX_RUNS"], "smoke_verdict.json"))); [print("  %-20s %s %s" % (k, "OK  " if x["ok"] else "DROP", x.get("why", ""))) for k, x in v.items()]'

step "llama arms (8 tasks x the arms smoke cleared, one server)" \
  uv run python benchmarks/matrix/run.py llama --from-smoke \
    --arms "$ARMS" --tasks "$TASKS" --timeout "$TIMEOUT" --keep
wait_server_down

# Two MLX arms: the shipping config, and the same engine with the DFlash2 drafter off.
# DFlash2 accepts by exact rejection sampling, so the pair isolates what speculation is
# worth in WALL CLOCK without changing the distribution the model writes from — which is
# the only way the engine cell stops measuring two things at once.
for arm in chad+mlx chad+mlx-nodflash; do
  for t in ${TASKS//,/ }; do
    step "$arm :: $t" uv run python benchmarks/matrix/run.py mlx \
      --arms "$arm" --tasks "$t" --timeout "$TIMEOUT" --keep
    sleep 45        # settle between model load/teardown cycles
  done
done

echo
echo "=== tables :: $(date '+%H:%M:%S') ==="
uv run python benchmarks/matrix/run.py table --tasks "$TASKS" 2>&1 | tee "$R/tables.md"
uv run python benchmarks/matrix/scorecard.py 2>&1 | tail -3
echo "=== done $(date '+%F %H:%M:%S') ==="
