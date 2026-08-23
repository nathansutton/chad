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
#   CHAD_MODEL_LABEL  harbor -m id — the submission's DISPLAYED model id (default
#                     Ornith-1.0-35B-Q6_K, a bare model name — the agent column
#                     already carries org=chad; cosmetic for --backend llama, no
#                     weights load from it)
#   CHAD_TB2_TEMP     sampling temperature (default 1.0, the reference recipe)
#   CHAD_TB2_MINP     min_p quant-tail filter (default 0.05; 0 = off)
#   TB21_NO_UPLOAD=1  skip --upload --public (local compliance dry-runs only —
#                     a real submission run MUST upload)
#
# Experiment knobs (all default-inert; a submission run behaves as if they did not
# exist). They exist so an experimental ARM runs through THIS script rather than a
# fork of it: the compliance rules and the six infra guards below are the expensive,
# hard-won part, and a config that was only ever measured under a private copy of
# them is not a config you can publish.
#   TB21_TASKS      comma list of task names, or a path to a file with one per line
#                   — restrict the run to that subset (e.g. a 12-task
#                   screen set). Unset = all 89, as before.
#   TB21_OUT        results tsv path (default tb21_submit_<stamp>.tsv here)
#   TB21_MAX_HOURS  stop LAUNCHING new tasks once this many hours have elapsed since
#                   the run started (the task in flight finishes and is scored). The
#                   tail truncates instead of overrunning the window; resume the rest
#                   later with RESUME_TSV. Unset = no guard, as before.
#   CHAD_PROJECT    chad checkout to upload into the containers, pinning the tree
#                   under test (default: the repo this script lives in). Also
#                   selects which harbor_chad_tb2.py is imported — the two are
#                   versioned together and a mismatch kills every trial at argparse.
# Args: REPEATS (harbor -k; default 5 — the leaderboard minimum)
#       TASK    (optional: run only this task, for the compliance smoke)
set -u
cd "$(dirname "$0")"
CHAD_PROJECT="${CHAD_PROJECT:-}"
if [ -n "$CHAD_PROJECT" ]; then
  [ -d "$CHAD_PROJECT/benchmarks/tb2" ] || { echo "FATAL: CHAD_PROJECT=$CHAD_PROJECT has no benchmarks/tb2"; exit 1; }
  export PYTHONPATH="$CHAD_PROJECT/benchmarks/tb2"
else
  export PYTHONPATH="$PWD"
fi

BASE_URL="${CHAD_BASE_URL:?set CHAD_BASE_URL to your model server as reachable from inside Docker}"
BACKEND="${CHAD_BACKEND:-llama}"
# Tokenizer stays the real MLX HF repo (nathansutton/…) — it's snapshot_download'd
# into the upload, so it must resolve; its vocab is quant-invariant. MODEL_LABEL is
# purely the leaderboard-displayed model id (cosmetic for --backend llama), so it reads
# org=chad, model=Ornith rather than surfacing the personal HF namespace as the "org".
TOKENIZER="${CHAD_TOKENIZER:-nathansutton/Ornith-1.0-35B-Q6-MLX}"
MODEL_LABEL="${CHAD_MODEL_LABEL:-Ornith-1.0-35B-Q6_K}"
TEMP="${CHAD_TB2_TEMP:-1.0}"
# Quant-tail filter: temp stays 1.0 but sub-noise-floor tail mass — a source of
# garbled-tool-call / foreign-dialect derailments the bf16 reference recipe never
# sampled — is cut. 0 turns it back off.
MINP="${CHAD_TB2_MINP:-0.05}"
REPEATS="${1:-5}"; ONLY="${2:-}"

DATASET="terminal-bench/terminal-bench-2-1"
DATASET_NAME="${DATASET#*/}"
TB21="${TB21_DATASET_DIR:-$PWD/dataset/$DATASET_NAME}"
if [ ! -d "$TB21" ]; then
  echo "=== downloading $DATASET dataset -> $PWD/dataset ==="
  harbor download "$DATASET" -o "$PWD/dataset" </dev/null || exit 1
fi

UPLOAD_FLAGS=(--upload --public)
if [ "${TB21_NO_UPLOAD:-0}" = "1" ]; then
  UPLOAD_FLAGS=()
  echo "WARN: TB21_NO_UPLOAD=1 — trials will NOT reach the Hub; fine for a dry run,"
  echo "      useless for a submission."
else
  harbor auth status </dev/null >/dev/null 2>&1 || { echo "FATAL: run 'harbor auth login' first"; exit 1; }
fi

HOST_URL="${BASE_URL/host.docker.internal/localhost}"
if ! curl -sf -m 5 "${HOST_URL%/v1}/health" >/dev/null 2>&1 \
   && ! curl -sf -m 5 "${HOST_URL%/}/models" >/dev/null 2>&1; then
  echo "WARN: no model server answering at $HOST_URL (host-side probe)"
fi

STAMP=$(date +%Y%m%d_%H%M%S)
# Resume: set RESUME_TSV=<prior tsv> to continue a run that died partway. We keep
# only rows that already carry a Hub job link (real, uploaded results) and strip
# every ERR/no-link row, so a task is re-run iff it never uploaded — and the
# dashboard tracks the same file cleanly as backfill lands.
RESUME_TSV="${RESUME_TSV:-}"
if [ -n "$RESUME_TSV" ]; then
  RESULTS="$RESUME_TSV"
  [ -f "$RESULTS" ] || { echo "FATAL: RESUME_TSV=$RESULTS not found"; exit 1; }
  tmp="${RESULTS}.tmp.$$"
  # An uploading run proves a row is real by its Hub link. A no-upload run (an
  # experimental arm) has no link to prove it with, so the completeness test there
  # is a numeric reward — ERR rows still get dropped and re-run either way.
  if [ "${TB21_NO_UPLOAD:-0}" = "1" ]; then
    awk -F'\t' 'NR==1 || $4 ~ /^[0-9]/' "$RESULTS" > "$tmp" && mv "$tmp" "$RESULTS"
  else
    awk -F'\t' 'NR==1 || $6 ~ /^https/' "$RESULTS" > "$tmp" && mv "$tmp" "$RESULTS"
  fi
  echo "=== RESUME into $RESULTS — $(($(grep -c . "$RESULTS")-1)) tasks already scored, will skip them ==="
else
  RESULTS="${TB21_OUT:-tb21_submit_${STAMP}.tsv}"
  mkdir -p "$(dirname "$RESULTS")"
  echo -e "task\tbudget\tchad_cap\treward\twall_s\tjob_link" > "$RESULTS"
fi
echo "=== TB2.1 submission run start $(date) backend=$BACKEND k=$REPEATS -> $RESULTS ==="

# Backend preflight, FROM A CONTAINER. Reachable-from-the-host is the wrong test:
# on 20260726 the Mac could curl thelio over the LAN fine while Docker Desktop's
# VM could not route to 192.168.87.0/24 at all (internet worked, LAN did not), so
# every trial died on a connection timeout, scored 0, and the run looked merely
# slow for two hours. Fail loudly in 30s instead. Best-effort: if the probe image
# can't be fetched, note it and go on rather than blocking the run.
if [ "$BACKEND" = "llama" ] && [ -n "${CHAD_BASE_URL:-}" ]; then
  probe=$(docker run --rm --add-host=host.docker.internal:host-gateway \
            curlimages/curl:latest -s -m 10 -o /dev/null \
            -w '%{http_code}' "${CHAD_BASE_URL%/}/health" 2>/dev/null || true)
  if [ "$probe" = "200" ]; then
    echo "=== preflight: containers reach $CHAD_BASE_URL (HTTP 200) ==="
  elif [ -z "$probe" ]; then
    echo "!! preflight SKIPPED (probe image unavailable) — backend unverified"
  else
    echo "FATAL: containers cannot reach $CHAD_BASE_URL (got '${probe:-no response}')."
    echo "       The host may still reach it while Docker's VM cannot route to the LAN."
    echo "       Relay it through the host and use http://host.docker.internal:PORT:"
    echo "         python3 ~/assist/private/lan_relay.py 8081 <server-ip> 8081 &"
    exit 1
  fi
fi

# Never-two-clients guard. llama-server runs --parallel 1: a second client does not
# queue politely, it halves every stream's decode rate against wall-clock task budgets
# and quietly converts capability fails into timeout fails in BOTH runs. The rule has
# always existed and has always been enforced by hand, which is why on 20260806 another
# arm took the box while BARE89 was down and BARE89 was never restarted.
#
# Ask the SERVER, not the process table: the old thelio-side check was `pgrep -f
# run_tb2.sh`, which could not work — it does not match this script's name, and chad
# runs on the Mac while the server runs on thelio, so no single host's process list can
# see both clients. These two metrics are host-agnostic:
#   requests_processing > 0   someone is mid-request right now
#   n_decode_total advancing  someone is streaming, even if every spot check landed in
#                             the gap between two of their turns
# Startup only — once we are the client, our own traffic would trip it. That leaves a
# residual race if another run starts during one of our env builds; a lockfile is the
# airtight version if that ever actually bites.
# TB21_SKIP_EXCLUSIVE=1 to override (deliberate co-tenancy, or a non-llama.cpp server).
exclusive_preflight() {
  [ "$BACKEND" = "llama" ] || return 0
  if [ "${TB21_SKIP_EXCLUSIVE:-0}" = "1" ]; then
    echo "!! exclusivity check SKIPPED (TB21_SKIP_EXCLUSIVE=1) — you are promising the server is yours"
    return 0
  fi
  local m="${HOST_URL%/v1}/metrics" a b busy i=0
  a=$(curl -sf -m 8 "$m" 2>/dev/null | awk '/^llamacpp:n_decode_total /{print $2}')
  if [ -z "$a" ]; then
    echo "=== exclusivity check skipped (no llama.cpp /metrics at $m) ==="
    return 0
  fi
  while :; do
    busy=0
    a=$(curl -sf -m 8 "$m" 2>/dev/null | awk '/^llamacpp:n_decode_total /{print $2}')
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      [ "$(curl -sf -m 8 "$m" 2>/dev/null | awk '/^llamacpp:requests_processing /{print $2}')" != "0" ] && busy=1
      sleep 4
    done
    b=$(curl -sf -m 8 "$m" 2>/dev/null | awk '/^llamacpp:n_decode_total /{print $2}')
    [ -n "$b" ] && [ "$a" != "$b" ] && busy=1
    if [ "$busy" = "0" ]; then
      [ $i -gt 0 ] && echo "=== server free $(date) — starting ==="
      echo "=== exclusivity: no other client on ${m%/metrics} ==="
      return 0
    fi
    i=$((i+1))
    [ $i -eq 1 ] && echo "!! another client is already using ${m%/metrics} $(date) — waiting for it to finish (never two clients on --parallel 1)"
    [ $i -gt 240 ] && { echo "FATAL: server still busy after 4h — aborting $(date)"; exit 1; }
    sleep 60
  done
}
exclusive_preflight

# Docker-dead guard. Twice now a mid-run infra death (stdin EBADF on 20260721,
# Docker backend SIGTERM'd on 20260722) made every remaining harbor call fail in
# ~1s and the loop burned the whole task list as ERR rows in two minutes. A task
# must never be scored ERR because the *host* environment is down — block here
# until Docker answers again (nudging Docker Desktop awake), up to 4h.
docker_wait() {
  docker info >/dev/null 2>&1 && return 0
  echo "!! Docker daemon down $(date) — waiting for it to return (relaunching Docker Desktop)"
  open -ga Docker 2>/dev/null || true
  local i=0
  while ! docker info >/dev/null 2>&1; do
    i=$((i+1))
    [ $i -gt 240 ] && { echo "FATAL: Docker still down after 4h — aborting run $(date)"; exit 1; }
    [ $((i % 30)) -eq 0 ] && open -ga Docker 2>/dev/null
    sleep 60
  done
  echo "!! Docker back $(date) — resuming"
}

# Backend-dead guard. The preflight above runs ONCE; nothing re-checked the model
# server mid-run. On 20260729 macOS re-prompted for Local Network access and the
# host->LAN relay started answering "No route to host" — every trial died on its
# FIRST engine call ("llama-server connection failed: RemoteDisconnected", 3
# retries, no tokens ever sampled) and harbor banked a clean 0.000. The
# environment-infra guard could not see it: the only string in harbor's output
# was "[chad-tb2] could not pull /tmp/chad.prefill.jsonl", which the grep -v on
# line ~267 deliberately ignores (chad never writes that trace when the backend
# is unreachable). 11 tasks false-zeroed over 90min before anyone noticed.
# Same contract as docker_wait: a task must never be scored because OUR host
# lost the model server. Probe FROM A CONTAINER — that is the path that breaks
# (on 20260726 the Mac could curl thelio while Docker's VM could not route).
backend_wait() {
  [ "$BACKEND" = "llama" ] && [ -n "${CHAD_BASE_URL:-}" ] || return 0
  local i=0 probe
  while :; do
    probe=$(docker run --rm --add-host=host.docker.internal:host-gateway \
              curlimages/curl:latest -s -m 10 -o /dev/null \
              -w '%{http_code}' "${CHAD_BASE_URL%/}/health" 2>/dev/null || true)
    # Empty = probe image unavailable, not a dead backend. Don't block on that.
    [ "$probe" = "200" ] || [ -z "$probe" ] && break
    i=$((i+1))
    [ $i -eq 1 ] && echo "!! backend unreachable at $CHAD_BASE_URL (got '$probe') $(date) — waiting"
    [ $i -gt 240 ] && { echo "FATAL: backend still unreachable after 4h — aborting run $(date)"; exit 1; }
    sleep 60
  done
  [ $i -gt 0 ] && echo "!! backend back $(date) — resuming"
  return 0
}

# --- run order --------------------------------------------------------------
# Default is alphabetical (deterministic, what the published repro does). Set
# TASK_ORDER_TSV=<a prior run's tsv> to walk the tasks INFORMATION-FIRST instead:
#   1. tasks that were flaky in the reference (0 < reward < 1), highest-variance
#      first — the band where a build change can actually show up
#   2. tasks the reference never scored (unknown)
#   3. tasks it always passed (regression watch)
#   4. tasks it always failed, quickest-first — these say the same thing every
#      run, so they are the ones to leave for last (and they are usually the
#      full-budget timeouts, so deferring them raises early tasks/hour too)
# Order changes nothing about scores: all 89 tasks still run at the same k.
# NOTE: with this on, the early overall% is biased LOW (the flaky band leads and
# the solid passes are deferred) — read the same-task delta, not the running %.
TASKS=()
while IFS= read -r d; do TASKS+=("$(basename "$d")"); done < <(ls -d "$TB21"/*/ | sort)
# TB21_TASKS: restrict to a named subset (a screen set, a tail-conversion
# list, ...). Every name must exist in the dataset — a typo'd task would otherwise
# silently shrink the arm and make it incomparable to its siblings, which is the
# same class of lie as a typo'd CHAD_DISABLE entry.
if [ -n "${TB21_TASKS:-}" ]; then
  if [ -f "$TB21_TASKS" ]; then
    want=$(grep -vE '^\s*(#|$)' "$TB21_TASKS" | tr -d ' \t\r')
  else
    want=$(printf '%s' "$TB21_TASKS" | tr ', ' '\n\n' | grep -v '^$')
  fi
  sel=()
  while IFS= read -r t; do
    [ -z "$t" ] && continue
    [ -d "$TB21/$t" ] || { echo "FATAL: TB21_TASKS names '$t', which is not a task in $TB21"; exit 1; }
    sel+=("$t")
  done <<< "$want"
  [ ${#sel[@]} -eq 0 ] && { echo "FATAL: TB21_TASKS resolved to zero tasks"; exit 1; }
  TASKS=("${sel[@]}")
  echo "=== task subset: ${#TASKS[@]} tasks from TB21_TASKS ==="
fi
if [ -n "${TASK_ORDER_TSV:-}" ]; then
  [ -f "$TASK_ORDER_TSV" ] || { echo "FATAL: TASK_ORDER_TSV=$TASK_ORDER_TSV not found"; exit 1; }
  ordered=$(printf '%s\n' "${TASKS[@]}" | awk -F'\t' -v ref="$TASK_ORDER_TSV" '
    BEGIN{ while ((getline line < ref) > 0) {
             n=split(line, f, "\t")
             if (n < 4 || f[1]=="task" || f[4]=="" || f[4]=="ERR") continue
             r[f[1]]=f[4]+0; w[f[1]]=f[5]+0; seen[f[1]]=1 } }
    { t=$1
      if (!(t in seen))          { tier=1; key=0 }
      else if (r[t]>0 && r[t]<1) { tier=0; key=-r[t]*(1-r[t]) }  # variance desc
      else if (r[t]>=1)          { tier=2; key=0 }
      else                       { tier=3; key=w[t] }            # cheapest first
      printf "%d\t%.6f\t%s\n", tier, key, t }' | sort -k1,1n -k2,2g -k3,3 | cut -f3)
  TASKS=(); while IFS= read -r t; do TASKS+=("$t"); done <<< "$ordered"
  echo "=== run order: information-first from $TASK_ORDER_TSV ==="
  echo "=== first 6: $(printf '%s ' "${TASKS[@]:0:6}")..."
fi

RUN_T0=$(date +%s)
for t in ${TASKS[@]+"${TASKS[@]}"}; do
  d="$TB21/$t"
  [ -n "$ONLY" ] && [ "$t" != "$ONLY" ] && continue
  if [ -n "${TB21_MAX_HOURS:-}" ] && \
     [ $(( $(date +%s) - RUN_T0 )) -ge $(awk -v h="$TB21_MAX_HOURS" 'BEGIN{printf "%d", h*3600}') ]; then
    echo "=== TB21_MAX_HOURS=$TB21_MAX_HOURS reached — not launching $t or anything after it (resume with RESUME_TSV=$RESULTS) ==="
    break
  fi
  # Resume skip: this task already landed a real row in a prior run of this same tsv.
  if [ -n "$RESUME_TSV" ]; then
    if [ "${TB21_NO_UPLOAD:-0}" = "1" ]; then
      awk -F'\t' -v t="$t" '$1==t && $4 ~ /^[0-9]/{f=1} END{exit !f}' "$RESULTS" \
        && { echo "skip $t (already scored)"; continue; }
    elif grep -qE "^${t}"$'\t'".*https://hub" "$RESULTS"; then
      echo "skip $t (already uploaded)"; continue
    fi
  fi
  budget=$(awk '/^\[agent\]/{a=1} a&&/timeout_sec/{gsub(/[^0-9.]/,"");print int($0);exit}' "$d/task.toml")
  [ -z "$budget" ] && budget=900
  cap=$((budget - 30))
  # Agent block goes through a job-config FILE, not --agent-import-path: the CLI
  # flag leaves config.agents[].name null, and the leaderboard's source_filter
  # check matches on that name ("no matching agent in job config" on the first
  # smoke). A config with BOTH name and import_path records name="chad" while
  # the factory still instantiates via the import path ("chad" is not a
  # registered harbor AgentName, so the registry branch falls through).
  mkdir -p "harbor_logs_${STAMP}"
  CFG="harbor_logs_${STAMP}/${t}.config.json"
  # CHAD_DISABLE ablation arm: export CHAD_DISABLE="lever,lever" before launching and
  # every trial in this run switches those levers off in-container (harbor_chad_tb2's
  # `chad_disable` kwarg). Omitted from the config entirely when unset, so a submission
  # run's job-config stays byte-identical to what it was before this knob existed.
  python3 - "$CFG" <<PYEOF
import json, os, sys
kwargs = {
    "chad_base_url": "$BASE_URL",
    "chad_backend": "$BACKEND",
    "chad_tokenizer": "$TOKENIZER",
    "chad_temp": "$TEMP",
    "chad_min_p": "$MINP",
    "chad_timeout_sec": $cap,
}
if os.environ.get("CHAD_DISABLE", "").strip():
    kwargs["chad_disable"] = os.environ["CHAD_DISABLE"].strip()
if os.environ.get("CHAD_PROJECT", "").strip():
    kwargs["chad_project"] = os.environ["CHAD_PROJECT"].strip()
json.dump({"agents": [{
    "name": "chad",
    "import_path": "harbor_chad_tb2:ChadAgent",
    "model_name": "$MODEL_LABEL",
    "kwargs": kwargs,
}]}, open(sys.argv[1], "w"), indent=1)
PYEOF
  # </dev/null: never let harbor inherit the launching terminal's stdin. If that
  # terminal/pty goes away mid-run (laptop sleep, ssh/tmux drop) fd 0 becomes a
  # closed descriptor, and CPython aborts every child at startup with
  # "init_sys_streams: can't initialize sys standard streams" (EBADF). That killed
  # 76/89 tasks of the 20260721 run. Reading from /dev/null makes the loop
  # terminal-independent.
  # Retry the task (not record ERR) when the failure is the host environment:
  # docker down before/after the attempt, or harbor saying so in its output.
  attempt=0
  while :; do
    docker_wait
    backend_wait
    t0=$(date +%s)
    # -n = trial-level concurrency (env build/setup/verify). Was $REPEATS (=k) so
    # all k trials built up front; dropped to 1 on 20260727 because overlapping
    # VERIFIERS were the top infra-failure source. Tasks pin `cpus = 1`, so five
    # concurrent verifiers contend for one core each against a 900-1800s verifier
    # timeout; the ML-heavy ones (mteb-retrieve, torch-{tensor,pipeline}-parallelism,
    # model-extraction-relu-logits) then blow the limit and score 0. Measured:
    # mteb-retrieve went 0 -> 1 -> 4 VerifierTimeoutErrors across three runs while
    # every agent exited rc=0. -n 1 gives each verifier an uncontended core.
    # This costs wall-clock (build/verify no longer overlap the agent phase) and
    # buys correctness — a deliberate trade, the host is NOT the pace bottleneck
    # (measured 20260727: agent containers sit at ~10% CPU, blocked on remote
    # decode from thelio, and freeing 7 wedged cores changed per-task pace by -6%).
    # --n-concurrent-agents 1 still serializes the AGENT phase: the llama.cpp
    # server has one slot, and two agents sharing it would halve each stream's
    # decode rate against wall-clock task budgets. Concurrency knobs are not
    # leaderboard-checked (the CI only rejects timeout multipliers and
    # cpu/gpu/mem/storage overrides).
    out=$(harbor run -c "$CFG" -d "$DATASET" -n 1 --n-concurrent-agents 1 \
      --include-task-name "${DATASET%%/*}/$t" \
      ${UPLOAD_FLAGS[@]+"${UPLOAD_FLAGS[@]}"} -k "$REPEATS" </dev/null 2>&1)
    reward=$(printf '%s' "$out" | grep -oiE "Mean: [01]\.[0-9]+" | tail -1 | grep -oE "[01]\.[0-9]+")
    job_link=$(printf '%s' "$out" | grep -oE "https://hub\.harborframework\.com/jobs/[a-f0-9-]+" | tail -1)
    wall=$(( $(date +%s) - t0 ))
    if [ -z "$reward" ] && { printf '%s' "$out" | grep -qi "docker daemon is not running" \
                             || ! docker info >/dev/null 2>&1; }; then
      attempt=$((attempt+1))
      [ $attempt -gt 5 ] && { echo "!! $t: docker-dead retry limit hit — aborting run"; exit 1; }
      echo "!! $t attempt $attempt died with Docker down — will re-run this task, not score it"
      continue
    fi
    # Environment-infra taint: a dying-but-answering Docker fails trials with
    # compose/AddTestsDir errors while `docker info` still returns OK, so the
    # branch above misses it and the job "completes" with reward 0 (that
    # false-zeroed extract-elf on 20260722, 5 trials dead in 130s). Any such
    # error in the job output means the score is not the model's — re-run the
    # whole task (up to 3 tries), then score as-is with a loud warning.
    # The `grep -v` is load-bearing: the agent's own artifact pull logs
    # "[chad-tb2] could not pull /tmp/chad.prefill.jsonl: Docker compose command
    # failed ..." whenever chad never wrote that optional trace file (it doesn't
    # when the backend is unreachable, so no engine call ever happens). That
    # best-effort copy failure is not an environment fault, but it matched the
    # pattern below and silently re-ran every task 4x — 2h of the 20260726 run
    # went into re-running one dead task instead of reporting the dead backend.
    if printf '%s' "$out" | grep -v 'could not pull' \
         | grep -qiE 'Docker compose command failed|AddTestsDirError|Failed to add tests directory'; then
      attempt=$((attempt+1))
      if [ $attempt -le 3 ]; then
        echo "!! $t attempt $attempt had environment-infra errors (compose/AddTestsDir) — re-running task, not scoring it"
        sleep 30
        continue
      fi
      echo "!! $t: infra errors persisted after 3 attempts — scoring anyway, INSPECT THIS ROW"
    fi
    # Errored-trial taint. A trial that never produced a verdict
    # (VerifierTimeoutError, RewardFileNotFoundError, RuntimeError, ...) is folded
    # into harbor's mean as reward 0, so an infra failure is indistinguishable from
    # a model failure in the tsv. Neither guard above sees it: `docker info` is
    # fine and the output carries no compose/AddTestsDir string. Measured damage —
    # mteb-retrieve banked 0.400 on 4 scored trials (1 timeout) on 20260726 and
    # 0.000 on ONE scored trial (4 timeouts) on 20260727, with all 10 agent runs
    # exiting rc=0. Neither number is chad's. A 4-trial row is also a k=5
    # compliance risk. Re-run the task (up to 3 tries), then score with a warning.
    # This is about not shipping OUR host's flakiness: the leaderboard itself
    # scores errored trials as 0 on purpose ("infra robustness is score"), and
    # this changes nothing about how a clean run is scored.
    resjson=$(printf '%s' "$out" | grep -oE 'jobs/[0-9_-]+/result\.json' | tail -1)
    nerr=0
    if [ -n "$resjson" ] && [ -f "$resjson" ]; then
      nerr=$(python3 -c 'import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print(int((d.get("stats") or {}).get("n_errored_trials") or 0))
except Exception:
    print(0)' "$resjson" 2>/dev/null || echo 0)
    fi
    if [ "${nerr:-0}" -gt 0 ]; then
      attempt=$((attempt+1))
      if [ $attempt -le 3 ]; then
        echo "!! $t attempt $attempt: $nerr errored trial(s) (verifier/reward infra, not the model) — re-running task, not scoring it"
        sleep 30
        continue
      fi
      echo "!! $t: $nerr errored trial(s) persisted after 3 attempts — scoring anyway, INSPECT THIS ROW"
    fi
    # Partial-job taint. A job that harbor abandons with trials still pending is
    # averaged over whatever finished and written to the tsv as if it were the full
    # k. Measured: PB01's bn-fit-modify on 20260802 banked reward 0.000 from
    # n_completed_trials=1, n_pending_trials=4 — a k=1 draw recorded as a k=5 score,
    # with no warning anywhere. A short row is also a k=5 compliance risk.
    ndone=""
    if [ -n "$resjson" ] && [ -f "$resjson" ]; then
      ndone=$(python3 -c 'import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print(int((d.get("stats") or {}).get("n_completed_trials") or 0))
except Exception:
    print("")' "$resjson" 2>/dev/null || echo "")
    fi
    if [ -n "$ndone" ] && [ "$ndone" -lt "$REPEATS" ]; then
      attempt=$((attempt+1))
      if [ $attempt -le 3 ]; then
        echo "!! $t attempt $attempt: only $ndone/$REPEATS trial(s) completed — a partial job is not a k=$REPEATS score, re-running task"
        backend_wait
        sleep 30
        continue
      fi
      echo "!! $t: only $ndone/$REPEATS trials completed after 3 attempts — scoring anyway, INSPECT THIS ROW"
    fi
    # Backend-dead taint. backend_wait covers a backend already down when the task
    # STARTS; this covers one that drops mid-task. A trial that ends with chad's
    # fatal "the remote backend stopped answering" banner is not a model failure —
    # it is the 20260729 relay outage and the 20260801 thelio wifi flap.
    #
    # Two corrections from the 20260801/02 damage, where this guard was present and
    # caught NOTHING while 53 trials died:
    #   1. It only read chad.session.log, but a trial that dies before its first
    #      completion writes ONE line there ("INFO TURN start") — the banner lands in
    #      chad.stdout.log. Both files are read now.
    #   2. It required EVERY trial to be dead, on the theory that one dead trial
    #      among live ones is genuine flakiness. It is not: BARE-0 banked
    #      build-cython-ext at 0.200 with 3/5 dead and PB01 banked pypi-server at
    #      0.000 with 4/5 dead. A mean over a partly-dead draw is not a measurement,
    #      so ANY dead trial re-runs the task.
    jobdir="${resjson%/result.json}"
    ndead=0; nseen=0
    if [ -n "$jobdir" ] && [ -d "$jobdir" ]; then
      for ad in "$jobdir"/*/agent; do
        [ -d "$ad" ] || continue
        nseen=$((nseen+1))
        grep -qE 'the remote backend stopped answering|llama-server connection failed' \
             "$ad/chad.stdout.log" "$ad/chad.session.log" 2>/dev/null \
          && ndead=$((ndead+1))
      done
    fi
    if [ "$nseen" -gt 0 ] && [ "$ndead" -gt 0 ]; then
      attempt=$((attempt+1))
      if [ $attempt -le 3 ]; then
        echo "!! $t attempt $attempt: $ndead/$nseen trial(s) died with the model server unreachable — not the model, re-running task"
        backend_wait
        sleep 30
        continue
      fi
      echo "!! $t: backend-dead trials persisted after 3 attempts — scoring anyway, INSPECT THIS ROW"
    fi
    break
  done
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
