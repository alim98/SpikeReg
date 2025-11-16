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
  echo "[launcher] Searching for existing checkpoints in $RUN_ROOT..."
  
  pretrain_epochs=$(grep -E '^\s*pretrain_epochs:' "$CONFIG" | sed 's/.*:\s*\([0-9]\+\).*/\1/' || echo "0")
  
  pretrained_model=""
  final_model=""
  latest_finetune_epoch=-1
  latest_finetune_ckpt=""
  latest_pretrain_epoch=-1
  latest_pretrain_ckpt=""
  
  # Sort directories by job ID (numeric) descending to prioritize latest runs
  # Extract numeric job IDs, sort them descending, then process in that order
  # Since we process in descending order, the first match is the latest
  for job_dir in $(find "$RUN_ROOT" -maxdepth 1 -type d ! -path "$RUN_ROOT" -exec basename {} \; | grep -E '^[0-9]+$' | sort -rn); do
    full_path="$RUN_ROOT/$job_dir"
    # Skip symlinks (like latest_job, latest_group) - only process actual directories
    if [[ -L "$full_path" ]]; then
      continue
    fi
    if [[ -d "$full_path/checkpoints" ]]; then
      # Resolve to absolute path to avoid symlink issues
      abs_job_dir=$(cd "$full_path" && pwd)
      if [[ -f "$abs_job_dir/checkpoints/final_model.pth" ]]; then
        # Only set if not already set (first match is latest due to descending sort)
        if [[ -z "$final_model" ]]; then
          final_model="$abs_job_dir/checkpoints/final_model.pth"
        fi
      fi
      if [[ -f "$abs_job_dir/checkpoints/pretrained_model.pth" ]]; then
        # Only set if not already set (first match is latest due to descending sort)
        if [[ -z "$pretrained_model" ]]; then
          pretrained_model="$abs_job_dir/checkpoints/pretrained_model.pth"
        fi
      fi
      
      ckpt=$(latest_ckpt_in "$abs_job_dir" || true)
      if [[ -n "$ckpt" ]]; then
        epoch=$(echo "$ckpt" | sed 's/.*model_epoch_\([0-9]\+\)\.pth/\1/')
        if [[ "$epoch" =~ ^[0-9]+$ ]]; then
          if (( epoch >= pretrain_epochs )); then
            if (( epoch > latest_finetune_epoch )); then
              latest_finetune_epoch=$epoch
              latest_finetune_ckpt="$ckpt"
            fi
          else
            if (( epoch > latest_pretrain_epoch )); then
              latest_pretrain_epoch=$epoch
              latest_pretrain_ckpt="$ckpt"
            fi
          fi
        fi
      fi
    fi
  done
  
  if [[ -n "$final_model" ]]; then
    echo "[launcher] Training complete! Found final_model.pth: $final_model"
    initial_ckpt=$(cd "$(dirname "$final_model")" && pwd)/final_model.pth
  elif [[ -n "$latest_finetune_ckpt" ]]; then
    echo "[launcher] Found finetune checkpoint from epoch $latest_finetune_epoch: $latest_finetune_ckpt"
    initial_ckpt=$(cd "$(dirname "$latest_finetune_ckpt")" && pwd)/$(basename "$latest_finetune_ckpt")
  elif [[ -n "$pretrained_model" ]]; then
    echo "[launcher] Pretrain complete! Found pretrained_model.pth: $pretrained_model"
    echo "[launcher] Starting finetune phase..."
    # Use absolute path to avoid symlink issues
    pretrained_dir=$(cd "$(dirname "$pretrained_model")" && pwd)
    converted_model_path="$pretrained_dir/converted_model.pth"
    if [[ -f "$converted_model_path" ]]; then
      initial_ckpt=$(cd "$(dirname "$converted_model_path")" && pwd)/converted_model.pth
      echo "[launcher] Using converted_model.pth for finetune: $initial_ckpt"
    else
      initial_ckpt=$(cd "$(dirname "$pretrained_model")" && pwd)/pretrained_model.pth
      echo "[launcher] Using pretrained_model.pth for finetune (will convert): $initial_ckpt"
    fi
  elif [[ -n "$latest_pretrain_ckpt" ]]; then
    echo "[launcher] Found pretrain checkpoint from epoch $latest_pretrain_epoch: $latest_pretrain_ckpt"
    initial_ckpt=$(cd "$(dirname "$latest_pretrain_ckpt")" && pwd)/$(basename "$latest_pretrain_ckpt")
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
      
      # Re-search for the best checkpoint to resume from (same logic as initial search)
      pretrain_epochs=$(grep -E '^\s*pretrain_epochs:' "$CONFIG" | sed 's/.*:\s*\([0-9]\+\).*/\1/' || echo "0")
      
      restart_ckpt=""
      pretrained_model=""
      final_model=""
      latest_finetune_epoch=-1
      latest_finetune_ckpt=""
      latest_pretrain_epoch=-1
      latest_pretrain_ckpt=""
      
      # Sort directories by job ID (numeric) descending to prioritize latest runs
      # Extract numeric job IDs, sort them descending, then process in that order
      # Since we process in descending order, the first match is the latest
      for job_dir in $(find "$RUN_ROOT" -maxdepth 1 -type d ! -path "$RUN_ROOT" -exec basename {} \; | grep -E '^[0-9]+$' | sort -rn); do
        full_path="$RUN_ROOT/$job_dir"
        # Skip symlinks (like latest_job, latest_group) - only process actual directories
        if [[ -L "$full_path" ]]; then
          continue
        fi
        if [[ -d "$full_path/checkpoints" ]]; then
          # Resolve to absolute path to avoid symlink issues
          abs_job_dir=$(cd "$full_path" && pwd)
          if [[ -f "$abs_job_dir/checkpoints/final_model.pth" ]]; then
            # Only set if not already set (first match is latest due to descending sort)
            if [[ -z "$final_model" ]]; then
              final_model="$abs_job_dir/checkpoints/final_model.pth"
            fi
          fi
          if [[ -f "$abs_job_dir/checkpoints/pretrained_model.pth" ]]; then
            # Only set if not already set (first match is latest due to descending sort)
            if [[ -z "$pretrained_model" ]]; then
              pretrained_model="$abs_job_dir/checkpoints/pretrained_model.pth"
            fi
          fi
          
          ckpt=$(latest_ckpt_in "$abs_job_dir" || true)
          if [[ -n "$ckpt" ]]; then
            epoch=$(echo "$ckpt" | sed 's/.*model_epoch_\([0-9]\+\)\.pth/\1/')
            if [[ "$epoch" =~ ^[0-9]+$ ]]; then
              if (( epoch >= pretrain_epochs )); then
                if (( epoch > latest_finetune_epoch )); then
                  latest_finetune_epoch=$epoch
                  latest_finetune_ckpt="$ckpt"
                fi
              else
                if (( epoch > latest_pretrain_epoch )); then
                  latest_pretrain_epoch=$epoch
                  latest_pretrain_ckpt="$ckpt"
                fi
              fi
            fi
          fi
        fi
      done
      
      if [[ -n "$final_model" ]]; then
        echo "[launcher] Training complete! Found final_model.pth: $final_model"
        restart_ckpt=$(cd "$(dirname "$final_model")" && pwd)/final_model.pth
      elif [[ -n "$latest_finetune_ckpt" ]]; then
        echo "[launcher] Resuming finetune from epoch $latest_finetune_epoch: $latest_finetune_ckpt"
        restart_ckpt=$(cd "$(dirname "$latest_finetune_ckpt")" && pwd)/$(basename "$latest_finetune_ckpt")
      elif [[ -n "$pretrained_model" ]]; then
        echo "[launcher] Pretrain complete! Using pretrained_model.pth for finetune: $pretrained_model"
        # Use absolute path to avoid symlink issues
        pretrained_dir=$(cd "$(dirname "$pretrained_model")" && pwd)
        converted_model_path="$pretrained_dir/converted_model.pth"
        if [[ -f "$converted_model_path" ]]; then
          restart_ckpt=$(cd "$(dirname "$converted_model_path")" && pwd)/converted_model.pth
        else
          restart_ckpt=$(cd "$(dirname "$pretrained_model")" && pwd)/pretrained_model.pth
        fi
      elif [[ -n "$latest_pretrain_ckpt" ]]; then
        echo "[launcher] Resuming pretrain from epoch $latest_pretrain_epoch: $latest_pretrain_ckpt"
        restart_ckpt=$(cd "$(dirname "$latest_pretrain_ckpt")" && pwd)/$(basename "$latest_pretrain_ckpt")
      elif [[ -n "$ckpt_path" ]]; then
        echo "[launcher] Using checkpoint from current job: $ckpt_path"
        restart_ckpt=$(cd "$(dirname "$ckpt_path")" && pwd)/$(basename "$ckpt_path")
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
