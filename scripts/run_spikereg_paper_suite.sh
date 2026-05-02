#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_spikereg_paper_suite.sh [--mode freeze|evaluate|evaluate-submit|configs|ablate|all] [--submit]

Default mode is all: freeze current checkpoints, evaluate them, and generate ablation configs.
Use --submit to actually launch the ablation jobs through launch_spikereg_training.sh.

Environment overrides:
  RUN_ROOT       Run directory containing existing jobs
  OUT_ROOT       Output directory for frozen links, eval JSON/CSV, generated configs
  ANN_CKPT       Current best ANN checkpoint
  SNN_CKPT       Current best SNN checkpoint
  CONFIG_BASE    Direct-displacement config
  CONFIG_SVF     SVF/diffeomorphic config
  PYTHON         Python executable
  DEVICE         Evaluation device, default cuda
  MAX_PAIRS      Evaluation pair limit for smoke tests, default 0/full set
  SKIP_HD95      Set 1 for quick evaluation without HD95
  T_VALUES       Space-separated T ablation values, default "1 2 4 6 8"
  THRESHOLDS     Space-separated threshold percentiles, default "50 75 90 95 99 99.5"
USAGE
}

MODE="all"
SUBMIT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"

RUN_ROOT="${RUN_ROOT:-/u/almik/SpikeReg/symlink/experiments3/SR/runs}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/nexus/posix0/MBR-neuralsystems/alim/experiments3/SR}"
OUT_ROOT="${OUT_ROOT:-$EXPERIMENT_ROOT/paper_suite/$(date +%Y%m%d-%H%M%S)}"
ANN_CKPT="${ANN_CKPT:-$RUN_ROOT/8289292/checkpoints/pretrained_model.pth}"
SNN_CKPT="${SNN_CKPT:-$RUN_ROOT/8329274/checkpoints/final_model.pth}"
CONFIG_BASE="${CONFIG_BASE:-$REPO_ROOT/configs/spikereg_l2r_config.yaml}"
CONFIG_SVF="${CONFIG_SVF:-$REPO_ROOT/configs/spikereg_l2r_svf_config.yaml}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${VENV_PATH:-/ptmp/almik/cephclr_venv_rocm63}/bin/python" ]]; then
    PYTHON="${VENV_PATH:-/ptmp/almik/cephclr_venv_rocm63}/bin/python"
  else
    PYTHON="python3"
  fi
fi
DEVICE="${DEVICE:-cuda}"
MAX_PAIRS="${MAX_PAIRS:-0}"
SKIP_HD95="${SKIP_HD95:-0}"
T_VALUES="${T_VALUES:-1 2 4 6 8}"
THRESHOLDS="${THRESHOLDS:-50 75 90 95 99 99.5}"

mkdir -p "$OUT_ROOT"/{frozen,eval,configs,logs}

freeze_checkpoints() {
  for ckpt in "$ANN_CKPT" "$SNN_CKPT"; do
    if [[ ! -f "$ckpt" ]]; then
      echo "Missing checkpoint: $ckpt" >&2
      exit 1
    fi
  done
  ln -sfn "$(realpath "$ANN_CKPT")" "$OUT_ROOT/frozen/best_ann_pretrained_model.pth"
  ln -sfn "$(realpath "$SNN_CKPT")" "$OUT_ROOT/frozen/best_snn_final_model.pth"
  sha256sum "$(realpath "$ANN_CKPT")" "$(realpath "$SNN_CKPT")" > "$OUT_ROOT/frozen/checkpoint_manifest.sha256"
  printf "Frozen checkpoint links:\n  %s\n  %s\n" \
    "$OUT_ROOT/frozen/best_ann_pretrained_model.pth" \
    "$OUT_ROOT/frozen/best_snn_final_model.pth"
  echo "Manifest: $OUT_ROOT/frozen/checkpoint_manifest.sha256"
}

evaluate_checkpoint() {
  local name="$1"
  local ckpt="$2"
  local model_type="$3"
  local config="$4"
  local extra=()
  if [[ "$SKIP_HD95" == "1" ]]; then
    extra+=(--skip-hd95)
  fi
  if [[ "$MAX_PAIRS" != "0" ]]; then
    extra+=(--max-pairs "$MAX_PAIRS")
  fi
  "$PYTHON" evaluate_oasis.py \
    --checkpoint "$ckpt" \
    --config "$config" \
    --dataset-format pkl \
    --model-type "$model_type" \
    --device "$DEVICE" \
    --output-json "$OUT_ROOT/eval/${name}.json" \
    --output-csv "$OUT_ROOT/eval/${name}.csv" \
    "${extra[@]}" \
    | tee "$OUT_ROOT/logs/eval_${name}.log"
}

run_evaluations() {
  evaluate_checkpoint "ann_same_unet" "$ANN_CKPT" "ann" "$CONFIG_BASE"
  evaluate_checkpoint "snn_t4_threshold50" "$SNN_CKPT" "snn" "$CONFIG_BASE"
  write_summary_table
}

submit_evaluation() {
  freeze_checkpoints
  echo "[paper-suite] submitting full evaluation job"
  sbatch \
    --export="ALL,OUT_ROOT=${OUT_ROOT},ANN_CKPT=${ANN_CKPT},SNN_CKPT=${SNN_CKPT},CONFIG_BASE=${CONFIG_BASE},DEVICE=${DEVICE},MAX_PAIRS=${MAX_PAIRS},SKIP_HD95=${SKIP_HD95}" \
    "$REPO_ROOT/eval_spikereg_paper.sh"
}

write_summary_table() {
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
}

make_configs() {
  "$PYTHON" - "$CONFIG_BASE" "$CONFIG_SVF" "$OUT_ROOT/configs" "$T_VALUES" "$THRESHOLDS" <<'PY'
import copy
import sys
from pathlib import Path
import yaml

base_path, svf_path, out_dir, t_values, thresholds = sys.argv[1:]
out = Path(out_dir)
out.mkdir(parents=True, exist_ok=True)
base = yaml.safe_load(Path(base_path).read_text())
svf = yaml.safe_load(Path(svf_path).read_text())
ts = [int(x) for x in t_values.split()]
ths = [float(x) for x in thresholds.split()]

def write(name, cfg):
    path = out / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(path)

def set_t(cfg, t):
    cfg["model"]["input_time_window"] = t
    cfg["model"]["encoder_time_windows"] = [t] * len(cfg["model"]["encoder_time_windows"])
    cfg["model"]["decoder_time_windows"] = [t] * len(cfg["model"]["decoder_time_windows"])

def set_threshold(cfg, th):
    cfg.setdefault("conversion", {})["threshold_percentile"] = th

for source_name, source in [("direct", base), ("svf", svf)]:
    for t in ts:
        for th in ths:
            cfg = copy.deepcopy(source)
            set_t(cfg, t)
            set_threshold(cfg, th)
            write(f"{source_name}_t{t}_thr{str(th).replace('.', 'p')}", cfg)

no_distill = copy.deepcopy(base)
set_t(no_distill, 4)
set_threshold(no_distill, 50)
no_distill["training"]["loss"]["distillation_weight"] = 0.0
write("direct_t4_thr50_no_distill", no_distill)

no_spike_reg = copy.deepcopy(base)
set_t(no_spike_reg, 4)
set_threshold(no_spike_reg, 50)
no_spike_reg["training"]["loss"]["spike_weight"] = 0.0
no_spike_reg["training"]["loss"]["spike_balance_weight"] = 0.0
write("direct_t4_thr50_no_spike_reg", no_spike_reg)

fixed_lif = copy.deepcopy(base)
set_t(fixed_lif, 4)
set_threshold(fixed_lif, 50)
fixed_lif["model"]["learnable_neurons"] = False
write("direct_t4_thr50_fixed_lif", fixed_lif)

no_cal = copy.deepcopy(base)
set_t(no_cal, 4)
no_cal.setdefault("conversion", {})["threshold_percentile"] = 0.0
no_cal["conversion"]["skip_calibration"] = True
write("direct_t4_no_threshold_calibration", no_cal)

scratch = copy.deepcopy(base)
set_t(scratch, 4)
set_threshold(scratch, 50)
scratch["training"]["pretrain"] = False
scratch["training"]["pretrain_epochs"] = 0
scratch["training"]["loss"]["distillation_weight"] = 0.0
write("snn_scratch_t4_thr50", scratch)
PY
}

submit_ablations() {
  local cfg name
  for cfg in "$OUT_ROOT"/configs/*.yaml; do
    name="$(basename "$cfg" .yaml)"
    echo "[paper-suite] ablation config: $cfg"
    if [[ "$SUBMIT" != "1" ]]; then
      if [[ "$name" == snn_scratch_* ]]; then
        echo "[paper-suite] dry run: RUN_TAG=paper_${name} ./launch_spikereg_training.sh --config \"$cfg\""
      else
        echo "[paper-suite] dry run: RUN_TAG=paper_${name} ./launch_spikereg_training.sh --config \"$cfg\" --start_from_checkpoint \"$ANN_CKPT\""
      fi
      continue
    fi
    if [[ "$name" == snn_scratch_* ]]; then
      RUN_TAG="paper_${name}" EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
        ./launch_spikereg_training.sh --config "$cfg"
    else
      RUN_TAG="paper_${name}" EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
        ./launch_spikereg_training.sh --config "$cfg" --start_from_checkpoint "$ANN_CKPT"
    fi
  done
}

case "$MODE" in
  freeze) freeze_checkpoints;;
  evaluate) freeze_checkpoints; run_evaluations;;
  evaluate-submit) submit_evaluation;;
  configs) make_configs;;
  ablate) make_configs; submit_ablations;;
  all) freeze_checkpoints; run_evaluations; make_configs; submit_ablations;;
  *) echo "Unknown mode: $MODE" >&2; usage; exit 2;;
esac

echo "Paper-suite output: $OUT_ROOT"
