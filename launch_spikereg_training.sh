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
      echo ""
      return
    fi
    
    # Priority order per user specification:
    # 1. pretrain_epoch_n.pth (latest pretrain epoch)
    # 2. pretrain_best_model.pth
    # 3. converted_model.pth
    # 4. finetune_epoch_n.pth (latest finetune epoch)
    
    # 1. Check for pretrain checkpoints (latest epoch)
    local pretrain_ckpt
    pretrain_ckpt=$(find "$ckpt_dir" -name 'pretrain_epoch_*.pth' 2>/dev/null | sort -V | tail -n1)
    if [[ -n "$pretrain_ckpt" ]]; then
      echo "$pretrain_ckpt"
      return
    fi
    
    # 2. Check for pretrain_best_model.pth
    if [[ -f "$ckpt_dir/pretrain_best_model.pth" ]]; then
      echo "$ckpt_dir/pretrain_best_model.pth"
      return
    fi
    
    # 3. Check for converted model
    if [[ -f "$ckpt_dir/converted_model.pth" ]]; then
      echo "$ckpt_dir/converted_model.pth"
      return
    fi
    
    # 4. Check for finetune checkpoints (latest epoch)
    local finetune_ckpt
    finetune_ckpt=$(find "$ckpt_dir" -name 'finetune_epoch_*.pth' 2>/dev/null | sort -V | tail -n1)
    if [[ -n "$finetune_ckpt" ]]; then
      echo "$finetune_ckpt"
      return
    fi
    
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

find_latest_checkpoint_recursive() {
  local search_root="$1"
  local best_ckpt=""
  local best_priority=999
  local best_epoch=-1
  
  # Search recursively for all checkpoints
  while IFS= read -r ckpt_path; do
    if [[ -z "$ckpt_path" ]]; then
      continue
    fi
    
    local priority=999
    local epoch=-1
    
    # Determine priority and epoch (lower priority number = higher priority)
    if [[ "$ckpt_path" == *"pretrain_epoch_"* ]]; then
      priority=1
      epoch=$(echo "$ckpt_path" | sed -E 's/.*pretrain_epoch_([0-9]+)\.pth/\1/')
    elif [[ "$ckpt_path" == *"pretrain_best_model.pth" ]]; then
      priority=2
    elif [[ "$ckpt_path" == *"converted_model.pth" ]]; then
      priority=3
    elif [[ "$ckpt_path" == *"finetune_epoch_"* ]]; then
      priority=4
      epoch=$(echo "$ckpt_path" | sed -E 's/.*finetune_epoch_([0-9]+)\.pth/\1/')
    else
      continue
    fi
    
    # Check if this checkpoint is better than current best
    if (( priority < best_priority )) || \
       (( priority == best_priority && epoch > best_epoch )); then
      best_ckpt="$ckpt_path"
      best_priority=$priority
      best_epoch=$epoch
    fi
  done < <(find "$search_root" -type f \( -name "finetune_epoch_*.pth" -o -name "converted_model.pth" -o -name "pretrain_best_model.pth" -o -name "pretrain_epoch_*.pth" \) 2>/dev/null)
  
  echo "$best_ckpt"
}

initial_ckpt=""
if [[ -n "$START_FROM" ]]; then
  initial_ckpt="$START_FROM"
else
  echo "[launcher] Searching for existing checkpoints in $RUN_ROOT and subfolders..."
  
  initial_ckpt=$(find_latest_checkpoint_recursive "$RUN_ROOT")
  
  if [[ -n "$initial_ckpt" ]]; then
    echo "[launcher] Found checkpoint: $initial_ckpt"
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
  
  # Check if training is actually complete before restarting
  training_complete=false
  if [[ "$state" == "COMPLETED" ]]; then
    # Check if final_model.pth exists (indicates all training phases completed)
    if [[ -f "$JOB_DIR/checkpoints/final_model.pth" ]]; then
      echo "[launcher] Training complete! final_model.pth exists in $JOB_DIR"
      training_complete=true
    elif [[ -f "$JOB_DIR/checkpoints/pretrained_model.pth" ]]; then
      # Pretrain is complete, check if we need to continue with finetune
      if [[ -f "$JOB_DIR/checkpoints/converted_model.pth" ]]; then
        echo "[launcher] Pretrain and conversion complete. Finetune should continue..."
      else
        echo "[launcher] Pretrain complete but conversion missing. Will convert and start finetune..."
      fi
    elif [[ -n "$ckpt_path" ]]; then
      # Extract epoch from checkpoint
      checkpoint_epoch=$(echo "$ckpt_path" | sed 's/.*model_epoch_\([0-9]\+\)\.pth/\1/')
      
      # Read pretrain epochs from config
      pretrain_epochs=$(grep -E '^\s*pretrain_epochs:' "$CONFIG" | sed 's/.*:\s*\([0-9]\+\).*/\1/' || echo "0")
      
      # If we've completed pretrain and final_model doesn't exist yet, continue (for finetune)
      # If checkpoint >= pretrain-1 but no final_model, finetune might be running
      if [[ "$checkpoint_epoch" =~ ^[0-9]+$ ]] && (( checkpoint_epoch >= pretrain_epochs - 1 )); then
        echo "[launcher] Pretrain complete (epoch $checkpoint_epoch >= $((pretrain_epochs-1))), checking for finetune completion..."
        # If no final_model after pretrain completion, it means finetune hasn't saved yet
        # This is actually the bug case - finetune might be stuck. But let's try one more restart.
      else
        echo "[launcher] Pretrain in progress. Checkpoint at epoch $checkpoint_epoch, target: $pretrain_epochs"
      fi
    fi
  fi
  
  # Only restart if training is not complete
  if [[ "$training_complete" == "true" ]]; then
    echo "[launcher] Training finished successfully. Stopping launcher."
    break
  fi
  
  case "$state" in
    COMPLETED|TIMEOUT|CANCELLED|FAILED|OUT_OF_MEMORY|PREEMPTED|NODE_FAIL)
      echo "[launcher] Job ended with state: $state. Restarting..."
      
      # Re-search for the best checkpoint to resume from (recursive search)
      restart_ckpt=$(find_latest_checkpoint_recursive "$RUN_ROOT")
      
      if [[ -n "$restart_ckpt" ]]; then
        echo "[launcher] Found checkpoint to resume from: $restart_ckpt"
        restart_ckpt=$(cd "$(dirname "$restart_ckpt")" && pwd)/$(basename "$restart_ckpt")
      elif [[ -n "$ckpt_path" ]]; then
        echo "[launcher] Using checkpoint from current job: $ckpt_path"
        restart_ckpt=$(cd "$(dirname "$ckpt_path")" && pwd)/$(basename "$ckpt_path")
      else
        echo "[launcher] No checkpoint found, starting from scratch"
        restart_ckpt=""
      fi
      
      sbatch_out=$(submit_job "$restart_ckpt" "$CONFIG")
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
