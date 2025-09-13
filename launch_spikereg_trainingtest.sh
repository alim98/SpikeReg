#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Repo root is the directory containing this script (SpikeReg)
REPO_ROOT="$SCRIPT_DIR"

JOB_SCRIPT="${JOB_SCRIPT:-$SCRIPT_DIR/train_spikeregtest.sh}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/checkpoints/test/runs}"
CONFIG_DEFAULT="${CONFIG_DEFAULT:-$REPO_ROOT/configs/spikereg_oasis_new_config.yaml}"

RUN_TAG="${RUN_TAG:-$(date +"%Y%m%d-%H%M%S")}"
GROUP_DIR="${RUN_ROOT}/${RUN_TAG}"
mkdir -p "$GROUP_DIR"
ln -sfn "$GROUP_DIR" "${RUN_ROOT}/latest_group"

START_FROM=""
CONFIG="$CONFIG_DEFAULT"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start_from_checkpoint) START_FROM="$2"; shift 2;;
    --config) CONFIG="$2"; shift 2;;
    *) EXTRA_ARGS+=("$1"); shift 1;;
  esac
done

latest_ckpt_in() {
  local dir="$1"
  if [[ -d "$dir/checkpoints" ]]; then
    find "$dir/checkpoints" -type f \( -name '*.pt' -o -name '*.pth' \) -printf "%T@ %p\n" | sort -nr | head -n1 | awk '{print $2}'
  else
    find "$dir" -type f -iname '*ckpt*' -printf "%T@ %p\n" | sort -nr | head -n1 | awk '{print $2}'
  fi
}

job_state() {
  local jid="$1"
  local s
  s=$(squeue -j "$jid" -h -o "%T" 2>/dev/null || true)
  if [[ -n "$s" ]]; then
    printf "%s\n" "$s"
    return 0
  fi
  s=$(sacct -j "$jid" -n -o State 2>/dev/null | head -n1 | awk '{print $1}')
  s="${s%%+*}"
  printf "%s\n" "$s"
}

submit_job() {
  local ckpt="${1:-}"
  local cfg="${2:-$CONFIG}"
  mkdir -p "$GROUP_DIR/logs"
  if [[ -n "$ckpt" ]]; then
    sbatch_out=$(sbatch "$JOB_SCRIPT" --config "$cfg" --start_from_checkpoint "$ckpt" "${EXTRA_ARGS[@]}")
  else
    sbatch_out=$(sbatch "$JOB_SCRIPT" --config "$cfg" "${EXTRA_ARGS[@]}")
  fi
  printf "%s\n" "$sbatch_out"
}

initial_ckpt=""
if [[ -n "$START_FROM" ]]; then
  initial_ckpt="$START_FROM"
fi

# Set test duration limit (1 hour = 3600 seconds)
TEST_START_TIME=$(date +%s)
TEST_DURATION_LIMIT=3600

sbatch_out=$(submit_job "$initial_ckpt" "$CONFIG")
echo "$sbatch_out"
job_id=$(echo "$sbatch_out" | awk '{print $4}')
echo "[launcher] Job ID: $job_id"
echo "[launcher] REPO_ROOT=$REPO_ROOT RUN_ROOT=$RUN_ROOT JOB_SCRIPT=$JOB_SCRIPT"
echo "[launcher] Test started at $(date), will run for maximum 1 hour"

while true; do
  # Check if we've exceeded the test duration
  current_time=$(date +%s)
  elapsed_time=$((current_time - TEST_START_TIME))
  if [[ $elapsed_time -gt $TEST_DURATION_LIMIT ]]; then
    echo "[launcher] Test duration limit reached (1 hour). Canceling current job and exiting."
    scancel "$job_id" 2>/dev/null || true
    echo "[launcher] Test completed after $((elapsed_time / 60)) minutes"
    break
  fi
  sleep 30
  state=$(job_state "$job_id")
  while [[ "$state" == "PENDING" || "$state" == "CONFIGURING" || "$state" == "RUNNING" || "$state" == "COMPLETING" ]]; do
    sleep 60
    state=$(job_state "$job_id")
  done
  echo "[launcher] Job $job_id finished with state: $state"

  JOB_DIR="${RUN_ROOT}/${job_id}"
  ln -sfn "$JOB_DIR" "$GROUP_DIR/last_jobdir"
  ln -sfn "$JOB_DIR" "${RUN_ROOT}/latest_job"

  ckpt_path="$(latest_ckpt_in "$JOB_DIR" || true)"
  case "$state" in
    COMPLETED|TIMEOUT|CANCELLED|FAILED|OUT_OF_MEMORY|PREEMPTED|NODE_FAIL)
      # Check time limit before submitting new job
      current_time=$(date +%s)
      elapsed_time=$((current_time - TEST_START_TIME))
      if [[ $elapsed_time -gt $TEST_DURATION_LIMIT ]]; then
        echo "[launcher] Test duration limit reached. Not submitting new job."
        echo "[launcher] Test completed after $((elapsed_time / 60)) minutes"
        break
      fi
      
      echo "[launcher] Submitting new job with checkpoint: ${ckpt_path:-none}"
      sbatch_out=$(submit_job "${ckpt_path:-}" "$CONFIG")
      echo "$sbatch_out"
      job_id=$(echo "$sbatch_out" | awk '{print $4}')
      echo "[launcher] New Job ID: $job_id"
      ;;
    *)
      echo "[launcher] Stopping loop (state: $state)."
      break
      ;;
  esac
done
