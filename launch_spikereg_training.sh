#!/usr/bin/env bash
set -euo pipefail

# --- USER TUNABLES ---
JOB_SCRIPT="train_spikereg.sh"
RUN_ROOT="checkpoints/spikereg/runs"
CONFIG_DEFAULT="configs/spikereg_oasis_new_config.yaml"   
EXTRA_ARGS=()                                     # put extra CLI args here or pass via CLI

# Group tag so all daily restarts sit in one folder
RUN_TAG=${RUN_TAG:-$(date +"%Y%m%d-%H%M%S")}
GROUP_DIR="${RUN_ROOT}/${RUN_TAG}"
mkdir -p "$GROUP_DIR"

# symlink for convenience
ln -sfn "$GROUP_DIR" "${RUN_ROOT}/latest_group"

# -------- helpers --------
function latest_ckpt_in() {
  local dir="$1"
  # Prefer .pt/.pth in 'checkpoints' subdir; fall back to any file with 'ckpt'
  if [[ -d "$dir/checkpoints" ]]; then
    find "$dir/checkpoints" -type f \( -name '*.pt' -o -name '*.pth' \) -printf "%T@ %p\n" \
      | sort -nr | head -n1 | awk '{print $2}'
  else
    find "$dir" -type f -iname '*ckpt*' -printf "%T@ %p\n" \
      | sort -nr | head -n1 | awk '{print $2}'
  fi
}

# Robust job-state: use squeue while active; sacct after finish
function job_state() {
  local jid="$1"
  local s
  s=$(squeue -j "$jid" -h -o "%T" 2>/dev/null || true)
  if [[ -n "$s" ]]; then
    echo "$s" && return 0
  fi
  sacct -j "$jid" -n -o State 2>/dev/null | head -n1 | awk '{print $1}'
}

# Submit a job. If $1 provided, treat it as --start_from_checkpoint path.
function submit_job() {
  local ckpt="${1:-}"
  local cfg="${2:-$CONFIG_DEFAULT}"

  mkdir -p "$GROUP_DIR/logs"
  if [[ -n "$ckpt" ]]; then
    echo "[launcher] Resuming from $ckpt"
    sbatch_out=$(sbatch "$JOB_SCRIPT" --config "$cfg" --start_from_checkpoint "$ckpt" "${EXTRA_ARGS[@]}")
  else
    echo "[launcher] Fresh start with config $cfg"
    sbatch_out=$(sbatch "$JOB_SCRIPT" --config "$cfg" "${EXTRA_ARGS[@]}")
  fi
  echo "$sbatch_out"
}

# -------- main loop --------
# 1) first submit
sbatch_out=$(submit_job "")
echo "$sbatch_out"
job_id=$(echo "$sbatch_out" | awk '{print $4}')
echo "[launcher] SpikeReg Job ID: $job_id"

# 2) monitor & restart forever until you Ctrl-C
while true; do
  sleep 30
  state=$(job_state "$job_id")
  # While queued/running, poll every minute
  while [[ "$state" == "PENDING" || "$state" == "CONFIGURING" || "$state" == "RUNNING" ]]; do
    sleep 60
    state=$(job_state "$job_id")
  done

  echo "[launcher] Job $job_id finished with state: $state"

  # Derive run dir of the finished job and find latest ckpt there
  JOB_DIR="${RUN_ROOT}/${job_id}"
  ln -sfn "$JOB_DIR" "$GROUP_DIR/last_jobdir"
  ln -sfn "$JOB_DIR" "${RUN_ROOT}/latest_job"

  ckpt_path=$(latest_ckpt_in "$JOB_DIR")
  if [[ -z "$ckpt_path" ]]; then
    echo "[launcher] No checkpoint found in $JOB_DIR. Will restart from scratch."
  fi

  case "$state" in
    COMPLETED|TIMEOUT|CANCELLED|FAILED|OUT_OF_MEMORY)
      echo "[launcher] Job finished with state: $state"
      if [[ -n "$ckpt_path" ]]; then
        echo "[launcher] Found checkpoint: $ckpt_path"
        echo "[launcher] Restarting SpikeReg from checkpoint…"
      else
        echo "[launcher] No checkpoint found, restarting from scratch…"
      fi
      sbatch_out=$(submit_job "$ckpt_path")
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
