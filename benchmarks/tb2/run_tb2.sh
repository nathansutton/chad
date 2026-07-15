#!/bin/bash
# Full Terminal-Bench 2.0 run for chad + Ornith-35B — the public reproduction script.
# See README.md in this directory for the whole recipe (serving the model, fidelity notes).
#
# It reads each task's [agent] timeout_sec from the dataset and gives chad (budget-30)s so
# chad's own catchable timeout fires just before harbor's hard cap — faithful per-task
# budgets AND a cleanly-flushed in-container stdout/trajectory on the timeout path.
#
# Required env:
#   CHAD_BASE_URL   model server origin AS SEEN FROM INSIDE THE TASK CONTAINERS,
#                   e.g. http://host.docker.internal:8080/v1 (chad --serve on this Mac)
#                   or   http://100.x.y.z:8081               (llama.cpp on a GPU box)
# Optional env:
#   CHAD_BACKEND      openai (default — a chat-completions server on the Mac running
#                     the benchmark; `chad --serve --host 0.0.0.0 --port 8080` is the
#                     recommended one: unlike mlx_lm.server it keeps the prefix KV
#                     cache warm across an agent's steps, so a warm step prefills
#                     ~tens of tokens instead of the whole transcript)
#                     or llama (llama.cpp raw /completion on a GPU box — the
#                     substitution our earlier reference runs used when 24 GB
#                     couldn't hold the 35B; plan 075 made the Mac arm primary)
#   CHAD_TOKENIZER    HF repo whose tokenizer matches the served model's vocab
#                     (default: nathansutton/Ornith-1.0-35B-UD-Q2_K_XL-MLX — the served
#                     MLX repo itself; Ornith's vocab is quant-invariant, so the same
#                     value is right for every Ornith-35B GGUF on the llama arm too)
#   CHAD_MODEL_LABEL  harbor -m run label (cosmetic — no weights load from it)
#   CHAD_TB2_TEMP     sampling temperature chad sends per-request (default 1.0, the
#                     TB2 reference recipe; top-p/k/min-p stay SERVER-side)
#   CHAD_TB2_THINK_CEILING  close-and-continue think ceiling (plan 086), e.g. 6000.
#                     Unset (default) = feature OFF, byte-identical to pre-086 chad.
#   TB2_DATASET_DIR   existing dataset export; downloaded here on first run otherwise
# Args: MAXCAP  (cap chad_timeout at this many sec, e.g. 1800 to bound wall-clock; 0 = uncapped)
#       REPEATS (harbor -k; default 1)
#       TASK    (optional: run only this task, e.g. `./run_tb2.sh 0 1 fix-git` to smoke one)
set -u
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"

BASE_URL="${CHAD_BASE_URL:?set CHAD_BASE_URL to your model server as reachable from inside Docker, e.g. http://host.docker.internal:8080/v1}"
BACKEND="${CHAD_BACKEND:-openai}"
TOKENIZER="${CHAD_TOKENIZER:-nathansutton/Ornith-1.0-35B-UD-Q2_K_XL-MLX}"
MODEL_LABEL="${CHAD_MODEL_LABEL:-mlx/nathansutton/Ornith-1.0-35B-UD-Q2_K_XL-MLX}"
TEMP="${CHAD_TB2_TEMP:-1.0}"
THINK_CEILING="${CHAD_TB2_THINK_CEILING:-}"
MAXCAP="${1:-0}"; REPEATS="${2:-1}"; ONLY="${3:-}"

# Dataset: harbor exports <dir>/terminal-bench-2/<task>/ — we need the task.toml files
# locally to read per-task [agent] budgets. ~1 GB on first download.
TB2="${TB2_DATASET_DIR:-$PWD/dataset/terminal-bench-2}"
if [ ! -d "$TB2" ]; then
  echo "=== downloading terminal-bench-2 dataset -> $PWD/dataset ==="
  harbor download terminal-bench/terminal-bench-2 -o "$PWD/dataset" || exit 1
fi

# Best-effort health probe: the container-visible URL may not resolve on the host
# (host.docker.internal doesn't), so probe a host-side equivalent and only warn.
# mlx_lm.server answers /v1/models; llama-server answers /health on its bare origin.
HOST_URL="${BASE_URL/host.docker.internal/localhost}"
if ! curl -sf -m 5 "${HOST_URL%/v1}/health" >/dev/null 2>&1 \
   && ! curl -sf -m 5 "${HOST_URL%/}/models" >/dev/null 2>&1; then
  echo "WARN: no model server answering at $HOST_URL (host-side probe) — make sure it is"
  echo "      up and that $BASE_URL is reachable from inside Docker containers."
fi

STAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="tb2_full_${STAMP}.tsv"
echo -e "task\tbudget\tchad_cap\treward\twall_s" > "$RESULTS"
echo "=== TB2 full run start $(date) backend=$BACKEND MAXCAP=$MAXCAP REPEATS=$REPEATS temp=$TEMP -> $RESULTS ==="
for d in "$TB2"/*/; do
  t=$(basename "$d")
  [ -n "$ONLY" ] && [ "$t" != "$ONLY" ] && continue
  budget=$(awk '/^\[agent\]/{a=1} a&&/timeout_sec/{gsub(/[^0-9.]/,"");print int($0);exit}' "$d/task.toml")
  [ -z "$budget" ] && budget=900
  cap=$((budget - 30))
  [ "$MAXCAP" -gt 0 ] && [ "$cap" -gt "$MAXCAP" ] && cap="$MAXCAP"
  t0=$(date +%s)
  extra_ak=()
  [ -n "$THINK_CEILING" ] && extra_ak+=(--ak "chad_think_ceiling=$THINK_CEILING")
  reward=$(harbor run -d terminal-bench/terminal-bench-2 --agent-import-path harbor_chad_tb2:ChadAgent \
    -m "$MODEL_LABEL" --n-concurrent-agents 1 \
    --agent-setup-timeout-multiplier 12 --environment-build-timeout-multiplier 3 \
    --include-task-name "terminal-bench/$t" \
    --ak chad_base_url="$BASE_URL" \
    --ak chad_backend="$BACKEND" \
    --ak chad_tokenizer="$TOKENIZER" \
    --ak chad_temp="$TEMP" --ak chad_timeout_sec=$cap "${extra_ak[@]}" -k "$REPEATS" 2>&1 \
    | grep -oiE "Mean: [01]\.[0-9]+" | tail -1 | grep -oE "[01]\.[0-9]+")
  wall=$(( $(date +%s) - t0 ))
  [ -z "$reward" ] && reward="ERR"
  echo -e "${t}\t${budget}\t${cap}\t${reward}\t${wall}" | tee -a "$RESULTS"
done
echo "=== TB2 full run DONE $(date) ==="
echo "--- tally ---"
awk -F'\t' 'NR>1 && $4!="ERR"{n++; s+=$4} NR>1&&$4=="ERR"{e++} END{printf "tasks=%d  pass_rate=%.1f%% (%.1f/%d)  errors=%d\n", n, (n?100*s/n:0), s, n, e+0}' "$RESULTS"
