#!/bin/bash -l
# --- Slurm header ---
#SBATCH -D ./
#SBATCH -o ./slurm/output_spikereg/%j.out
#SBATCH -e ./slurm/error_spikereg/%j.err
#SBATCH -J spikereg_training
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4           # one Python/DDP rank per GPU
#SBATCH -A mhf_gpu
#SBATCH --qos=g0008
#SBATCH --cpus-per-task=16
#SBATCH --mem=125G
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --mail-type=NONE
#SBATCH --time=23:00:00

set -euo pipefail

# --- Modules/conda (match your cluster) ---
module purge
module load anaconda/3/2023.03
module load cuda/11.6
module load gcc/11
module load openmpi_gpu/4.1

# Activate the conda environment
source /mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/etc/profile.d/conda.sh
conda activate "${SPIKEREG_ENV:-cephclr}"

# Load PyTorch distributed module (must load after anaconda module is loaded)
module load pytorch-distributed/gpu-cuda-11.6/2.1.0

# --- Python & performance knobs ---
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export NCCL_DEBUG=WARN
export PYTHON=${PYTHON_PATH:-$(which python)}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
export SPIKEREG_KEEP_LAST="${SPIKEREG_KEEP_LAST:-5}"

# --- Aim repo (optional): default to repo-level path alongside checkpoints ---
export AIM_REPO="${AIM_REPO:-${REPO_ROOT}/checkpoints/spikereg}"

# Ensure Aim repo exists and is initialized
mkdir -p "$AIM_REPO"
if [[ ! -d "$AIM_REPO/.aim" ]]; then
  echo "[train_spikereg] Initializing Aim repo at $AIM_REPO"
  aim init --repo "$AIM_REPO" | cat
fi

# --- Training entrypoint (adjust if your CLI differs) ---
# Common patterns:
#   python -m spikereg.train --config ... [--resume path]
#   python SpikeReg/training.py --config ... [--resume path]
# --- Run as a MODULE, not a script path ---
TRAIN_ENTRY_MODULE=${TRAIN_ENTRY_MODULE:-"SpikeReg.training"}
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TOTAL_GPUS=$(( SLURM_NNODES * GPUS_PER_NODE ))

MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
MASTER_PORT="${MASTER_PORT:-$((15000 + SLURM_JOB_ID % 20000))}"
export MASTER_ADDR MASTER_PORT

set -x
# Build arguments array
TRAIN_ARGS=(
  --name "${SLURM_JOB_ID}"
  --config "$CONFIG"
  --checkpoint_dir "$SPIKEREG_CHECKPOINT_DIR"
  --log_dir "$RUNDIR/logs"
  --aim_repo "$AIM_REPO"
)

# Add checkpoint resumption if specified
if [[ -n "$START_FROM" ]]; then
  TRAIN_ARGS+=(--start_from_checkpoint "$START_FROM")
fi

# Add any other arguments
TRAIN_ARGS+=("${OTHER_ARGS[@]}")

srun \
  --ntasks="$TOTAL_GPUS" \
  --ntasks-per-node="$GPUS_PER_NODE" \
  --kill-on-bad-exit=1 \
  "$PYTHON" -m "$TRAIN_ENTRY_MODULE" \
  "${TRAIN_ARGS[@]}" \
  |& tee "$RUNDIR/logs/train_${SLURM_JOB_ID}.log"
set +x


# Keep a copy of the last used config next to logs/checkpoints
cp -f "$CONFIG" "$RUNDIR/training_config.yaml" || true
