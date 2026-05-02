#!/bin/bash -l
#SBATCH -D ./
#SBATCH -o ./slurm/output_spikereg/%j_eval.out
#SBATCH -e ./slurm/error_spikereg/%j_eval.err
#SBATCH -J spikereg_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=apu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120000
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

OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT to a paper_suite output directory}"
ANN_CKPT="${ANN_CKPT:?Set ANN_CKPT}"
SNN_CKPT="${SNN_CKPT:?Set SNN_CKPT}"
CONFIG_BASE="${CONFIG_BASE:?Set CONFIG_BASE}"
DEVICE="${DEVICE:-cuda}"
MAX_PAIRS="${MAX_PAIRS:-0}"
SKIP_HD95="${SKIP_HD95:-0}"

mkdir -p "$OUT_ROOT/eval" "$OUT_ROOT/logs"

extra=()
if [[ "$SKIP_HD95" == "1" ]]; then
  extra+=(--skip-hd95)
fi
if [[ "$MAX_PAIRS" != "0" ]]; then
  extra+=(--max-pairs "$MAX_PAIRS")
fi

run_eval() {
  local name="$1"
  local ckpt="$2"
  local model_type="$3"
  "$PYTHON" evaluate_oasis.py \
    --checkpoint "$ckpt" \
    --config "$CONFIG_BASE" \
    --dataset-format pkl \
    --model-type "$model_type" \
    --device "$DEVICE" \
    --output-json "$OUT_ROOT/eval/${name}.json" \
    --output-csv "$OUT_ROOT/eval/${name}.csv" \
    "${extra[@]}" \
    | tee "$OUT_ROOT/logs/eval_${name}_${SLURM_JOB_ID}.log"
}

run_eval "ann_same_unet" "$ANN_CKPT" "ann"
run_eval "snn_t4_threshold50" "$SNN_CKPT" "snn"

"$PYTHON" - "$OUT_ROOT/eval" <<'PY'
import csv
import json
import sys
from pathlib import Path

eval_dir = Path(sys.argv[1])
rows = []
for path in sorted(eval_dir.glob("*.json")):
    payload = json.loads(path.read_text())
    s = payload["summary"]
    rows.append({
        "method": path.stem,
        "dice": s.get("dice_mean"),
        "hd95": s.get("hd95_mean"),
        "ncc": s.get("ncc_mean"),
        "neg_jac_percent": 100.0 * s.get("jacobian_negative_fraction_mean", 0.0),
        "sdlogj": s.get("sdlogj_mean"),
        "mean_spike_rate": s.get("mean_spike_rate"),
        "ann_macs_G": s.get("ann_macs_G"),
        "snn_acs_G": s.get("snn_acs_G"),
        "energy_ratio_snn_over_ann": s.get("energy_ratio_snn_over_ann"),
        "energy_reduction_factor": s.get("energy_reduction_factor"),
        "inference_time_sec": s.get("inference_time_sec_mean"),
        "parameter_count": s.get("parameter_count"),
    })

out = eval_dir / "summary_table.csv"
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["method"])
    writer.writeheader()
    writer.writerows(rows)
print(out)
PY
