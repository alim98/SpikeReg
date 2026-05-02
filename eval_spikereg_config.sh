#!/bin/bash -l
#SBATCH -D ./
#SBATCH -o ./slurm/output_spikereg/%j_eval_config.out
#SBATCH -e ./slurm/error_spikereg/%j_eval_config.err
#SBATCH -J spikereg_eval_cfg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=apu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100000
#SBATCH --qos=n0064
#SBATCH --mail-type=NONE
#SBATCH --time=08:00:00

set -euo pipefail

VENV_PATH="${VENV_PATH:-/ptmp/almik/cephclr_venv_rocm63}"

module purge
module load gcc/14 rocm/6.3 openmpi_gpu/5.0

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHON="${PYTHON:-$VENV_PATH/bin/python}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

CONFIG="${CONFIG:?Set CONFIG to the YAML config to evaluate}"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT to a paper_suite output directory}"
ANN_CKPT="${ANN_CKPT:?Set ANN_CKPT}"
NAME="${EVAL_NAME:-$(basename "$CONFIG" .yaml)}"
DEVICE="${DEVICE:-cuda}"
MAX_PAIRS="${MAX_PAIRS:-0}"
SKIP_HD95="${SKIP_HD95:-0}"
CALIBRATION_PAIRS="${CALIBRATION_PAIRS:-3}"

mkdir -p "$OUT_ROOT/eval_ablation" "$OUT_ROOT/logs"

extra=()
if [[ "$SKIP_HD95" == "1" ]]; then
  extra+=(--skip-hd95)
fi
if [[ "$MAX_PAIRS" != "0" ]]; then
  extra+=(--max-pairs "$MAX_PAIRS")
fi

"$PYTHON" evaluate_oasis.py \
  --checkpoint "$ANN_CKPT" \
  --config "$CONFIG" \
  --dataset-format pkl \
  --convert-ann-to-snn \
  --calibration-pairs "$CALIBRATION_PAIRS" \
  --device "$DEVICE" \
  --output-json "$OUT_ROOT/eval_ablation/${NAME}.json" \
  --output-csv "$OUT_ROOT/eval_ablation/${NAME}.csv" \
  "${extra[@]}" \
  | tee "$OUT_ROOT/logs/eval_ablation_${NAME}_${SLURM_JOB_ID}.log"
