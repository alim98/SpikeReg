#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Repo root is the directory containing this script (SpikeReg)
REPO_ROOT="$SCRIPT_DIR"

JOB_SCRIPT="${JOB_SCRIPT:-$SCRIPT_DIR/train_spikereg.sh}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/checkpoints/spikereg/runs}"
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
    local ckpt_dir="$dir/checkpoints"
    
    if [[ ! -d "$ckpt_dir" ]]; then
      echo ""  # No checkpoints directory
      return
    fi
    
    # Find epoch-based checkpoints ONLY (highest epoch number)
    local epoch_ckpt
    epoch_ckpt=$(find "$ckpt_dir" -name 'model_epoch_*.pth' -type f | \
      sed 's/.*model_epoch_\([0-9]\+\)\.pth/\1 &/' | \
      sort -nr | head -n1 | awk '{print $2}')
    
    if [[ -n "$epoch_ckpt" ]]; then
      echo "$epoch_ckpt"
      return
    fi
    
    # No epoch checkpoints found
    echo ""
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
    echo "[launcher] Submitting job with checkpoint: $ckpt" >&2
    sbatch_out=$(sbatch "$JOB_SCRIPT" --config "$cfg" --start_from_checkpoint "$ckpt" "${EXTRA_ARGS[@]}")
  else
    echo "[launcher] Submitting job without checkpoint (fresh start)" >&2
    sbatch_out=$(sbatch "$JOB_SCRIPT" --config "$cfg" "${EXTRA_ARGS[@]}")
  fi
  printf "%s\n" "$sbatch_out"
}

initial_ckpt=""
if [[ -n "$START_FROM" ]]; then
  initial_ckpt="$START_FROM"
else
  # Search for the latest checkpoint across ALL job directories
  echo "[launcher] Searching for existing checkpoints in $RUN_ROOT..."
  latest_epoch=-1
  latest_ckpt=""
  
  for job_dir in "$RUN_ROOT"/*/; do
    if [[ -d "$job_dir/checkpoints" ]]; then
      ckpt=$(latest_ckpt_in "$job_dir" || true)
      if [[ -n "$ckpt" ]]; then
        # Extract epoch number
        epoch=$(echo "$ckpt" | sed 's/.*model_epoch_\([0-9]\+\)\.pth/\1/')
        if [[ "$epoch" =~ ^[0-9]+$ ]] && (( epoch > latest_epoch )); then
          latest_epoch=$epoch
          latest_ckpt="$ckpt"
        fi
      fi
    fi
  done
  
  if [[ -n "$latest_ckpt" ]]; then
    echo "[launcher] Found checkpoint from epoch $latest_epoch: $latest_ckpt"
    initial_ckpt="$latest_ckpt"
  else
    echo "[launcher] No existing checkpoints found, starting from scratch"
  fi
fi

sbatch_out=$(submit_job "$initial_ckpt" "$CONFIG")
echo "$sbatch_out"
job_id=$(echo "$sbatch_out" | grep -o '[0-9]\+$')
echo "[launcher] Extracted Job ID: $job_id"
echo "[launcher] REPO_ROOT=$REPO_ROOT RUN_ROOT=$RUN_ROOT JOB_SCRIPT=$JOB_SCRIPT"

while true; do
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

  # Look for checkpoints in the current job directory (which just finished)
  ckpt_path="$(latest_ckpt_in "$JOB_DIR" || true)"
  if [[ -n "$ckpt_path" ]]; then
    echo "[launcher] Found checkpoint: $ckpt_path"
  else
    echo "[launcher] No checkpoint found in $JOB_DIR"
  fi
  case "$state" in
    COMPLETED|TIMEOUT|CANCELLED|FAILED|OUT_OF_MEMORY|PREEMPTED|NODE_FAIL)
      sbatch_out=$(submit_job "${ckpt_path:-}" "$CONFIG")
      echo "$sbatch_out"
      job_id=$(echo "$sbatch_out" | grep -o '[0-9]\+$')
      echo "[launcher] Extracted New Job ID: $job_id"
      ;;
    *)
      echo "[launcher] Stopping loop (state: $state)."
      break
      ;;
  esac
done
