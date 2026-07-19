#!/bin/bash
# Terminal-Bench 2.1 LEADERBOARD-COMPLIANT runner for chad + Ornith.
#
# Same shape as run_tb2.sh (per-task jobs, faithful per-task [agent] budgets read
# from task.toml, chad capped at budget-30s) with the submission rules applied.
# The leaderboard CI (terminal-bench-2-1/leaderboard/src/leaderboard/ci/
# static_analysis.py) rejects, on the job config AND per-trial records:
#   - timeout_multiplier != None/1.0
#   - ANY agent/verifier/setup/build timeout multiplier or override
#     (run_tb2.sh's --agent-setup-timeout-multiplier 12 and
#     --environment-build-timeout-multiplier 3 are instant rejections — this
#     script passes NEITHER; chad's whole setup must fit harbor's hard 360s
#     default, which it does: p90 18s measured, and the tokenizer now ships
#     inside the upload so setup never touches the HF Hub)
#   - any environment resource override
# and requires: 89 tasks x >=5 trials, every rewarded trial carrying an ATIF
# trajectory_path on the Hub. Errored trials count reward 0 — they are NOT
# excluded, so infra robustness is score.
#
# Trials upload to the Harbor Hub (PUBLIC — the leaderboard requires it) as they
# finish; `harbor auth login` first. Submit afterwards from a clone of
# https://github.com/harbor-framework/terminal-bench-2-1:
#     cd leaderboard && uv run lb submit <hub job links...>
#
# Required env:
#   CHAD_BASE_URL   model server origin AS SEEN FROM INSIDE THE TASK CONTAINERS
#                   (http://host.docker.internal:8080 for a Mac-local server,
#                    a LAN IP like http://<host>:8081 for thelio llama.cpp)
# Optional env (defaults are the Q6 submission arm):
#   CHAD_BACKEND      llama                     (default llama — thelio Q6_K)
#   CHAD_TOKENIZER    HF repo for the tokenizer (default the Q6 MLX repo; Ornith's
#                     vocab is quant-invariant so it matches every Ornith artifact)
#   CHAD_MODEL_LABEL  harbor -m id — becomes the submission's model id, so name
#                     the artifact actually served
#   CHAD_TB2_TEMP     sampling temperature (default 1.0, the reference recipe)
#   TB21_NO_UPLOAD=1  skip --upload --public (local compliance dry-runs only —
#                     a real submission run MUST upload)
# Args: REPEATS (harbor -k; default 5 — the leaderboard minimum)
#       TASK    (optional: run only this task, for the compliance smoke)
set -u
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"

BASE_URL="${CHAD_BASE_URL:?set CHAD_BASE_URL to your model server as reachable from inside Docker}"
BACKEND="${CHAD_BACKEND:-llama}"
TOKENIZER="${CHAD_TOKENIZER:-nathansutton/Ornith-1.0-35B-Q6-MLX}"
MODEL_LABEL="${CHAD_MODEL_LABEL:-mlx/nathansutton/Ornith-1.0-35B-Q6-MLX}"
TEMP="${CHAD_TB2_TEMP:-1.0}"
REPEATS="${1:-5}"; ONLY="${2:-}"

DATASET="terminal-bench/terminal-bench-2-1"
DATASET_NAME="${DATASET#*/}"
TB21="${TB21_DATASET_DIR:-$PWD/dataset/$DATASET_NAME}"
if [ ! -d "$TB21" ]; then
  echo "=== downloading $DATASET dataset -> $PWD/dataset ==="
  harbor download "$DATASET" -o "$PWD/dataset" || exit 1
fi

UPLOAD_FLAGS=(--upload --public)
if [ "${TB21_NO_UPLOAD:-0}" = "1" ]; then
  UPLOAD_FLAGS=()
  echo "WARN: TB21_NO_UPLOAD=1 — trials will NOT reach the Hub; fine for a dry run,"
  echo "      useless for a submission."
else
  harbor auth status >/dev/null 2>&1 || { echo "FATAL: run 'harbor auth login' first"; exit 1; }
fi

HOST_URL="${BASE_URL/host.docker.internal/localhost}"
if ! curl -sf -m 5 "${HOST_URL%/v1}/health" >/dev/null 2>&1 \
   && ! curl -sf -m 5 "${HOST_URL%/}/models" >/dev/null 2>&1; then
  echo "WARN: no model server answering at $HOST_URL (host-side probe)"
fi

STAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="tb21_submit_${STAMP}.tsv"
echo -e "task\tbudget\tchad_cap\treward\twall_s\tjob_link" > "$RESULTS"
echo "=== TB2.1 submission run start $(date) backend=$BACKEND k=$REPEATS -> $RESULTS ==="
for d in "$TB21"/*/; do
  t=$(basename "$d")
  [ -n "$ONLY" ] && [ "$t" != "$ONLY" ] && continue
  budget=$(awk '/^\[agent\]/{a=1} a&&/timeout_sec/{gsub(/[^0-9.]/,"");print int($0);exit}' "$d/task.toml")
  [ -z "$budget" ] && budget=900
  cap=$((budget - 30))
  t0=$(date +%s)
  # Agent block goes through a job-config FILE, not --agent-import-path: the CLI
  # flag leaves config.agents[].name null, and the leaderboard's source_filter
  # check matches on that name ("no matching agent in job config" on the first
  # smoke). A config with BOTH name and import_path records name="chad" while
  # the factory still instantiates via the import path ("chad" is not a
  # registered harbor AgentName, so the registry branch falls through).
  mkdir -p "harbor_logs_${STAMP}"
  CFG="harbor_logs_${STAMP}/${t}.config.json"
  python3 - "$CFG" <<PYEOF
import json, sys
json.dump({"agents": [{
    "name": "chad",
    "import_path": "harbor_chad_tb2:ChadAgent",
    "model_name": "$MODEL_LABEL",
    "kwargs": {
        "chad_base_url": "$BASE_URL",
        "chad_backend": "$BACKEND",
        "chad_tokenizer": "$TOKENIZER",
        "chad_temp": "$TEMP",
        "chad_timeout_sec": $cap,
    },
}]}, open(sys.argv[1], "w"), indent=1)
PYEOF
  out=$(harbor run -c "$CFG" -d "$DATASET" --n-concurrent-agents 1 \
    --include-task-name "${DATASET%%/*}/$t" \
    "${UPLOAD_FLAGS[@]}" -k "$REPEATS" 2>&1)
  reward=$(printf '%s' "$out" | grep -oiE "Mean: [01]\.[0-9]+" | tail -1 | grep -oE "[01]\.[0-9]+")
  job_link=$(printf '%s' "$out" | grep -oE "https://hub\.harborframework\.com/jobs/[a-f0-9-]+" | tail -1)
  wall=$(( $(date +%s) - t0 ))
  [ -z "$reward" ] && reward="ERR"
  # Keep harbor's full output — the first smoke lost an upload failure ("claim
  # your Harbor username") because only reward/link were extracted from $out.
  mkdir -p "harbor_logs_${STAMP}"
  printf '%s\n' "$out" > "harbor_logs_${STAMP}/${t}.log"
  if [ ${#UPLOAD_FLAGS[@]} -gt 0 ] && [ -z "$job_link" ]; then
    echo "!! UPLOAD MISSING for $t (no hub link in harbor output) — tail:"
    printf '%s\n' "$out" | tail -5
    echo "!! recover with: harbor upload jobs/<jobdir> --public"
  fi
  echo -e "${t}\t${budget}\t${cap}\t${reward}\t${wall}\t${job_link:-none}" | tee -a "$RESULTS"
done
echo "=== TB2.1 submission run DONE $(date) ==="
awk -F'\t' 'NR>1 && $4!="ERR"{n++; s+=$4} NR>1&&$4=="ERR"{e++} END{printf "tasks=%d  pass_rate=%.1f%% (%.1f/%d)  errors=%d\n", n, (n?100*s/n:0), s, n, e+0}' "$RESULTS"
echo "job links for lb submit are in column 6 of $RESULTS"
