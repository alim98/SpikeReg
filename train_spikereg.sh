#!/bin/bash -l
# --- Slurm header ---
#SBATCH -D ./
#SBATCH -o ./slurm/output_spikereg/%j.out
#SBATCH -e ./slurm/error_spikereg/%j.err
#SBATCH -J spikereg_training
#SBATCH --nodes=4
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=apu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=220000
#SBATCH --qos=n0064
#SBATCH --mail-type=NONE
#SBATCH --time=23:00:00

set -euo pipefail

VENV_PATH="${VENV_PATH:-/ptmp/almik/cephclr_venv_rocm63}"

module purge
module load gcc/14 rocm/6.3 openmpi_gpu/5.0

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export OMP_WAIT_POLICY=PASSIVE
export OMP_PLACES=cores
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHON="${VENV_PATH}/bin/python"

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
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/nexus/posix0/MBR-neuralsystems/alim/experiments3/SR}"
RUNDIR="${RUNDIR:-$EXPERIMENT_ROOT/runs/${SLURM_JOB_ID}}"
mkdir -p "$RUNDIR/logs" "$RUNDIR/checkpoints"
ln -sfn "$RUNDIR" "$EXPERIMENT_ROOT/runs/latest_job"

# --- Safety: checkpoint often (so restart has fresh state) ---
export SPIKEREG_CHECKPOINT_DIR="${RUNDIR}/checkpoints"
export SPIKEREG_KEEP_LAST="${SPIKEREG_KEEP_LAST:-5}"

export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${RUNDIR}/tensorboard}"
mkdir -p "$TENSORBOARD_DIR"

mkdir -p "/ptmp/almik/miopen_cache/${SLURM_JOB_ID}"

RANK_WRAPPER="${RUNDIR}/rank_wrapper.sh"
cat > "$RANK_WRAPPER" << 'WRAPPER'
#!/bin/bash
export MIOPEN_USER_DB_PATH="/ptmp/almik/miopen_cache/${SLURM_JOB_ID}/rank_${SLURM_PROCID}"
export MIOPEN_CACHE_DIR="/ptmp/almik/miopen_cache/${SLURM_JOB_ID}/rank_${SLURM_PROCID}"
mkdir -p "$MIOPEN_USER_DB_PATH"
exec "$@"
WRAPPER
chmod +x "$RANK_WRAPPER"

TRAIN_ENTRY_MODULE=${TRAIN_ENTRY_MODULE:-"SpikeReg.training"}
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
TOTAL_GPUS=$(( SLURM_NNODES * GPUS_PER_NODE ))

MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
MASTER_PORT="${MASTER_PORT:-$((15000 + SLURM_JOB_ID % 20000))}"
export MASTER_ADDR MASTER_PORT

set -x
TRAIN_ARGS=(
  --name "${SLURM_JOB_ID}"
  --config "$CONFIG"
  --checkpoint_dir "$SPIKEREG_CHECKPOINT_DIR"
  --log_dir "$RUNDIR/logs"
  --tensorboard_dir "$TENSORBOARD_DIR"
)

if [[ -n "$START_FROM" ]]; then
  TRAIN_ARGS+=(--start_from_checkpoint "$START_FROM")
fi

TRAIN_ARGS+=("${OTHER_ARGS[@]}")

srun \
  --ntasks="$TOTAL_GPUS" \
  --ntasks-per-node="$GPUS_PER_NODE" \
  --kill-on-bad-exit=1 \
  "$RANK_WRAPPER" "$PYTHON" -m "$TRAIN_ENTRY_MODULE" \
  "${TRAIN_ARGS[@]}" \
  |& tee "$RUNDIR/logs/train_${SLURM_JOB_ID}.log"
set +x


# Keep a copy of the last used config next to logs/checkpoints
cp -f "$CONFIG" "$RUNDIR/training_config.yaml" || true
