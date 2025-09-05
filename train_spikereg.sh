#!/bin/bash -l
# --- Slurm header ---
#SBATCH -D ./
#SBATCH -o ./slurm/output_spikereg/%j.out
#SBATCH -J spikereg_training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2           # # of processes (DDP)
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --mail-type=NONE
#SBATCH --time=24:00:00

set -euo pipefail

# --- Modules/conda (match your cluster) ---
module purge
module load intel/21.2.0 impi/2021.2

# If your SpikeReg lives in same env as VGG, keep cephclr. Otherwise set SPK_ENV.
source /mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/etc/profile.d/conda.sh
conda activate "${SPK_ENV:-spikereg}" || conda activate cephclr

# --- Python & performance knobs ---
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=WARN
export PYTHON=${PYTHON_PATH:-$(which python)}

# --- Args from sbatch line ---
CONFIG=""
START_FROM=""
OTHER_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --start_from_checkpoint) START_FROM="$2"; shift 2;;
    *) OTHER_ARGS+=("$1"); shift;;
  esac
done

if [[ -z "${CONFIG}" ]]; then
  echo "ERROR: --config <file.yaml> is required." >&2
  exit 1
fi

# --- Repository root & PYTHONPATH ---
# Prefer SLURM_SUBMIT_DIR (the directory where sbatch was invoked),
# fall back to the directory containing this script
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# --- Run dir & logging (absolute paths to avoid cwd confusion) ---
RUNDIR="$REPO_ROOT/checkpoints/spikereg/runs/${SLURM_JOB_ID}"
mkdir -p "$RUNDIR/logs" "$RUNDIR/checkpoints"
ln -sfn "$RUNDIR" "$REPO_ROOT/checkpoints/spikereg/runs/latest_job"

# --- Safety: checkpoint often (so restart has fresh state) ---
export SPIKEREG_CHECKPOINT_DIR="${RUNDIR}/checkpoints"
export SPIKEREG_CHECKPOINT_EVERY_STEPS="${SPIKEREG_CHECKPOINT_EVERY_STEPS:-1000}"   # tune
export SPIKEREG_KEEP_LAST="${SPIKEREG_KEEP_LAST:-5}"

# --- Training entrypoint (adjust if your CLI differs) ---
# Common patterns:
#   python -m spikereg.train --config ... [--resume path]
#   python SpikeReg/training.py --config ... [--resume path]
# --- Run as a MODULE, not a script path ---
TRAIN_ENTRY_MODULE=${TRAIN_ENTRY_MODULE:-"SpikeReg.training"}
# Use a single process; SpikeRegTrainer handles DataParallel internally if enabled
NP=1

set -x
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NP" \
  --module "$TRAIN_ENTRY_MODULE" \
  -- \
  --name "${SLURM_JOB_ID}" \
  --config "$CONFIG" \
  --checkpoint_dir "$SPIKEREG_CHECKPOINT_DIR" \
  --log_dir "$RUNDIR/logs" \
  "${OTHER_ARGS[@]}" \
  |& tee "$RUNDIR/logs/train_${SLURM_JOB_ID}.log"
set +x


# Keep a copy of the last used config next to logs/checkpoints
cp -f "$CONFIG" "$RUNDIR/training_config.yaml" || true
