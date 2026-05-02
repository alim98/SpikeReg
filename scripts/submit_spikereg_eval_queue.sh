#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/submit_spikereg_eval_queue.sh [options]

Evaluation-only ablation queue. It submits one Slurm job per YAML config,
keeps up to N jobs active, and each job converts the frozen ANN checkpoint to
SNN with that config's T/threshold/SVF settings, evaluates it, and exits.
No training or fine-tuning is performed.

Options:
  --config-dir DIR       Directory containing *.yaml configs. If omitted, configs
                         are generated under OUT_ROOT/configs.
  --out-root DIR         Paper-suite output directory.
  --max-jobs N           Number of simultaneous Slurm jobs. Default: 8.
  --poll-sec N           Poll interval in seconds. Default: 45.
  --detach               Start the queue manager in the background with nohup.
  --dry-run              Print submissions without calling sbatch.
  -h, --help             Show this help.

Environment overrides:
  EXPERIMENT_ROOT        Default: /nexus/posix0/MBR-neuralsystems/alim/experiments3/SR
  RUN_ROOT               Default: $EXPERIMENT_ROOT/runs
  ANN_CKPT               Default: current best ANN checkpoint
  EVAL_JOB_SCRIPT        Default: ./eval_spikereg_config.sh
  SBATCH_ACCOUNT         Default: mhf_apu
  SBATCH_PARTITION       Default: apu
  EVAL_MEM               Memory per one-GPU eval job. Default: 100000
  EVAL_TIME              Time limit per eval job. Default: 08:00:00
  EVAL_CPUS_PER_TASK     CPUs per eval job. Default: 12
  EVAL_QOS               Slurm QOS. Default: n0064
  MAX_JOBS               Default: 8
  MAX_PAIRS              Evaluation pair limit; 0 means all pairs
  SKIP_HD95              Set 1 to skip HD95 for faster smoke tests
  CALIBRATION_PAIRS      Calibration pairs for conversion. Default: 3
  DEVICE                 Default: cuda
  T_VALUES, THRESHOLDS   Passed to config generator if configs are generated
  INCLUDE_TRAINING_ONLY_EVALS
                         Set 1 to include configs whose differences only
                         matter during training. Default: 0

Outputs:
  $OUT_ROOT/eval_queue/queue.tsv
  $OUT_ROOT/eval_queue/submitted.tsv
  $OUT_ROOT/eval_queue/status.tsv
  $OUT_ROOT/eval_queue/submitter.log
  $OUT_ROOT/eval_ablation/*.json
  $OUT_ROOT/eval_ablation/*.csv
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/nexus/posix0/MBR-neuralsystems/alim/experiments3/SR}"
RUN_ROOT="${RUN_ROOT:-$EXPERIMENT_ROOT/runs}"
OUT_ROOT="${OUT_ROOT:-$EXPERIMENT_ROOT/paper_suite/$(date +%Y%m%d-%H%M%S)}"
ANN_CKPT="${ANN_CKPT:-$RUN_ROOT/8289292/checkpoints/pretrained_model.pth}"
EVAL_JOB_SCRIPT="${EVAL_JOB_SCRIPT:-$REPO_ROOT/eval_spikereg_config.sh}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-mhf_apu}"
SBATCH_PARTITION="${SBATCH_PARTITION:-apu}"
EVAL_MEM="${EVAL_MEM:-100000}"
EVAL_TIME="${EVAL_TIME:-08:00:00}"
EVAL_CPUS_PER_TASK="${EVAL_CPUS_PER_TASK:-12}"
EVAL_QOS="${EVAL_QOS:-n0064}"
MAX_JOBS="${MAX_JOBS:-8}"
POLL_SEC="${POLL_SEC:-45}"
MAX_PAIRS="${MAX_PAIRS:-0}"
SKIP_HD95="${SKIP_HD95:-0}"
CALIBRATION_PAIRS="${CALIBRATION_PAIRS:-3}"
DEVICE="${DEVICE:-cuda}"
INCLUDE_TRAINING_ONLY_EVALS="${INCLUDE_TRAINING_ONLY_EVALS:-0}"
CONFIG_DIR=""
DRY_RUN=0
DETACH=0
ORIGINAL_ARGS=("$@")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-dir) CONFIG_DIR="$2"; shift 2;;
    --out-root) OUT_ROOT="$2"; shift 2;;
    --max-jobs) MAX_JOBS="$2"; shift 2;;
    --poll-sec) POLL_SEC="$2"; shift 2;;
    --detach) DETACH=1; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2;;
  esac
done

if [[ ! -f "$EVAL_JOB_SCRIPT" ]]; then
  echo "Missing EVAL_JOB_SCRIPT: $EVAL_JOB_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$ANN_CKPT" ]]; then
  echo "Missing ANN_CKPT: $ANN_CKPT" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT" "$OUT_ROOT/eval_queue" "$OUT_ROOT/eval_ablation" "$OUT_ROOT/logs"
QUEUE_DIR="$OUT_ROOT/eval_queue"
LOG="$QUEUE_DIR/submitter.log"
QUEUE_TSV="$QUEUE_DIR/queue.tsv"
SUBMITTED_TSV="$QUEUE_DIR/submitted.tsv"
STATUS_TSV="$QUEUE_DIR/status.tsv"

if [[ "$DETACH" == "1" && "${SPIKEREG_EVAL_QUEUE_DETACHED:-0}" != "1" ]]; then
  forward_args=()
  for arg in "${ORIGINAL_ARGS[@]}"; do
    [[ "$arg" == "--detach" ]] && continue
    forward_args+=("$arg")
  done
  export SPIKEREG_EVAL_QUEUE_DETACHED=1
  nohup "$0" "${forward_args[@]}" > "$QUEUE_DIR/manager.nohup.log" 2>&1 &
  pid="$!"
  printf '%s\n' "$pid" > "$QUEUE_DIR/manager.pid"
  echo "Started eval queue manager PID $pid"
  echo "Log: $LOG"
  echo "Nohup log: $QUEUE_DIR/manager.nohup.log"
  echo "Output: $OUT_ROOT"
  exit 0
fi

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"
}

if [[ -z "$CONFIG_DIR" ]]; then
  log "Generating configs under $OUT_ROOT"
  OUT_ROOT="$OUT_ROOT" "$REPO_ROOT/scripts/run_spikereg_paper_suite.sh" --mode configs | tee -a "$LOG"
  CONFIG_DIR="$OUT_ROOT/configs"
fi

mapfile -t CONFIGS < <(
  find "$CONFIG_DIR" -maxdepth 1 -type f -name '*.yaml' | sort | while read -r cfg; do
    name="$(basename "$cfg" .yaml)"
    if [[ "$INCLUDE_TRAINING_ONLY_EVALS" != "1" ]]; then
      case "$name" in
        snn_scratch_*|*_no_distill|*_no_spike_reg|*_fixed_lif)
          continue
          ;;
      esac
    fi
    printf '%s\n' "$cfg"
  done
)
if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
  echo "No YAML configs found in $CONFIG_DIR" >&2
  exit 1
fi

printf 'index\tname\tconfig\n' > "$QUEUE_TSV"
for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  name="$(basename "$cfg" .yaml)"
  printf '%s\t%s\t%s\n' "$i" "$name" "$cfg" >> "$QUEUE_TSV"
done

printf 'job_id\tindex\tname\tconfig\tsubmitted_at\n' > "$SUBMITTED_TSV"
printf 'job_id\tindex\tname\tstate\tfinished_at\n' > "$STATUS_TSV"

declare -A ACTIVE_NAME=()
declare -A ACTIVE_INDEX=()
declare -A DONE=()
next_index=0

sbatch_state() {
  local jid="$1"
  local state
  state="$(squeue -j "$jid" -h -o "%T" 2>/dev/null | head -n1 || true)"
  if [[ -n "$state" ]]; then
    printf '%s\n' "$state"
    return
  fi
  state="$(sacct -j "$jid" -n -P -o State 2>/dev/null | head -n1 | cut -d'|' -f1 || true)"
  state="${state%%+*}"
  printf '%s\n' "${state:-UNKNOWN}"
}

is_active_state() {
  case "$1" in
    PENDING|CONFIGURING|RUNNING|COMPLETING|SUSPENDED|REQUEUED|RESIZING) return 0;;
    *) return 1;;
  esac
}

active_count() {
  printf '%s\n' "${!ACTIVE_NAME[@]}" | sed '/^$/d' | wc -l
}

submit_one() {
  local idx="$1"
  local cfg="${CONFIGS[$idx]}"
  local name
  name="$(basename "$cfg" .yaml)"

  local sbatch_args=()
  [[ -n "$SBATCH_ACCOUNT" ]] && sbatch_args+=(-A "$SBATCH_ACCOUNT")
  [[ -n "$SBATCH_PARTITION" ]] && sbatch_args+=(-p "$SBATCH_PARTITION")
  sbatch_args+=(
    --nodes 1
    --ntasks 1
    --gres gpu:1
    --cpus-per-task "$EVAL_CPUS_PER_TASK"
    --mem "$EVAL_MEM"
    --time "$EVAL_TIME"
    --qos "$EVAL_QOS"
    -J "sre_${name:0:18}"
  )

  local export_vars="ALL,OUT_ROOT=${OUT_ROOT},ANN_CKPT=${ANN_CKPT},CONFIG=${cfg},EVAL_NAME=${name},DEVICE=${DEVICE},MAX_PAIRS=${MAX_PAIRS},SKIP_HD95=${SKIP_HD95},CALIBRATION_PAIRS=${CALIBRATION_PAIRS}"
  local cmd=(sbatch "${sbatch_args[@]}" --export="$export_vars" "$EVAL_JOB_SCRIPT")

  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run submit [$idx/$(( ${#CONFIGS[@]} - 1 ))] $name: ${cmd[*]}"
    return
  fi

  log "submit [$idx/$(( ${#CONFIGS[@]} - 1 ))] $name"
  local out jid
  out="$("${cmd[@]}")"
  printf '%s\n' "$out" | tee -a "$LOG"
  jid="$(printf '%s\n' "$out" | grep -o '[0-9]\+$' | tail -n1)"
  if [[ -z "$jid" ]]; then
    log "ERROR: could not parse sbatch job id for $name"
    exit 1
  fi
  ACTIVE_NAME["$jid"]="$name"
  ACTIVE_INDEX["$jid"]="$idx"
  printf '%s\t%s\t%s\t%s\t%s\n' "$jid" "$idx" "$name" "$cfg" "$(date '+%F %T')" >> "$SUBMITTED_TSV"
}

write_summary_table() {
  python3 - "$OUT_ROOT/eval_ablation" <<'PY'
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
        "energy_ratio_snn_over_ann": s.get("energy_ratio_snn_over_ann"),
        "energy_reduction_factor": s.get("energy_reduction_factor"),
        "inference_time_sec": s.get("inference_time_sec_mean"),
    })
if rows:
    out = eval_dir / "summary_table.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out)
PY
}

if [[ "$DRY_RUN" == "1" ]]; then
  log "Dry run: ${#CONFIGS[@]} eval configs, max jobs $MAX_JOBS"
  for i in "${!CONFIGS[@]}"; do
    submit_one "$i"
  done
  log "Dry run complete"
  exit 0
fi

log "Evaluation queue started: ${#CONFIGS[@]} configs, max jobs $MAX_JOBS, poll ${POLL_SEC}s"
log "Config dir: $CONFIG_DIR"

while :; do
  for jid in "${!ACTIVE_NAME[@]}"; do
    state="$(sbatch_state "$jid")"
    if ! is_active_state "$state"; then
      name="${ACTIVE_NAME[$jid]}"
      idx="${ACTIVE_INDEX[$jid]}"
      log "finished job=$jid name=$name state=$state"
      printf '%s\t%s\t%s\t%s\t%s\n' "$jid" "$idx" "$name" "$state" "$(date '+%F %T')" >> "$STATUS_TSV"
      DONE["$jid"]="$state"
      unset 'ACTIVE_NAME[$jid]' 'ACTIVE_INDEX[$jid]'
    fi
  done

  while [[ "$next_index" -lt "${#CONFIGS[@]}" ]] && [[ "$(active_count)" -lt "$MAX_JOBS" ]]; do
    submit_one "$next_index"
    next_index=$((next_index + 1))
  done

  if [[ "$next_index" -ge "${#CONFIGS[@]}" ]] && [[ "$(active_count)" -eq 0 ]]; then
    break
  fi

  log "progress submitted=$next_index/${#CONFIGS[@]} active=$(active_count) done=${#DONE[@]}"
  sleep "$POLL_SEC"
done

write_summary_table | tee -a "$LOG"
log "Evaluation queue complete: submitted=${#CONFIGS[@]} done=${#DONE[@]}"
