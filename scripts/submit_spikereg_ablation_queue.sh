#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/submit_spikereg_ablation_queue.sh [options]

Submits SpikeReg ablation configs to Slurm and keeps up to N jobs active.
When one job leaves the Slurm queue, the next queued config is submitted until
all configs have run.

Options:
  --config-dir DIR       Directory containing *.yaml configs. If omitted, configs
                         are generated under OUT_ROOT/configs.
  --out-root DIR         Paper-suite output directory. Defaults to a timestamped
                         directory under EXPERIMENT_ROOT/paper_suite.
  --max-jobs N           Number of simultaneous Slurm jobs. Default: 8.
  --poll-sec N           Poll interval in seconds. Default: 60.
  --dry-run              Print submissions without calling sbatch.
  -h, --help             Show this help.

Environment overrides:
  EXPERIMENT_ROOT        Default: /nexus/posix0/MBR-neuralsystems/alim/experiments3/SR
  RUN_ROOT               Default: $EXPERIMENT_ROOT/runs
  ANN_CKPT               Default: current best ANN checkpoint
  JOB_SCRIPT             Default: ./train_spikereg.sh
  SBATCH_ACCOUNT         Default: mhf_apu
  SBATCH_PARTITION       Default: apu
  QUEUE_NODES            Slurm nodes per task. Default: 1
  QUEUE_GPUS_PER_NODE    GPUs per task/node. Default: 2
  QUEUE_NTASKS_PER_NODE  Slurm tasks per node. Default: $QUEUE_GPUS_PER_NODE
  QUEUE_CPUS_PER_TASK    CPUs per Slurm task. Default: 24
  QUEUE_MEM              Memory per job. Default: 220000
  QUEUE_TIME             Time limit per job. Default: 23:00:00
  QUEUE_QOS              Slurm QOS. Default: n0064
  MAX_JOBS               Default: 8
  POLL_SEC               Default: 60
  T_VALUES               Passed to config generator if configs are generated
  THRESHOLDS             Passed to config generator if configs are generated

Outputs:
  $OUT_ROOT/slurm_queue/queue.tsv
  $OUT_ROOT/slurm_queue/submitted.tsv
  $OUT_ROOT/slurm_queue/status.tsv
  $OUT_ROOT/slurm_queue/submitter.log
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/nexus/posix0/MBR-neuralsystems/alim/experiments3/SR}"
RUN_ROOT="${RUN_ROOT:-$EXPERIMENT_ROOT/runs}"
OUT_ROOT="${OUT_ROOT:-$EXPERIMENT_ROOT/paper_suite/$(date +%Y%m%d-%H%M%S)}"
ANN_CKPT="${ANN_CKPT:-$RUN_ROOT/8289292/checkpoints/pretrained_model.pth}"
JOB_SCRIPT="${JOB_SCRIPT:-$REPO_ROOT/train_spikereg.sh}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-mhf_apu}"
SBATCH_PARTITION="${SBATCH_PARTITION:-apu}"
QUEUE_NODES="${QUEUE_NODES:-1}"
QUEUE_GPUS_PER_NODE="${QUEUE_GPUS_PER_NODE:-2}"
QUEUE_NTASKS_PER_NODE="${QUEUE_NTASKS_PER_NODE:-$QUEUE_GPUS_PER_NODE}"
QUEUE_NTASKS="${QUEUE_NTASKS:-$(( QUEUE_NODES * QUEUE_NTASKS_PER_NODE ))}"
QUEUE_CPUS_PER_TASK="${QUEUE_CPUS_PER_TASK:-24}"
QUEUE_MEM="${QUEUE_MEM:-220000}"
QUEUE_TIME="${QUEUE_TIME:-23:00:00}"
QUEUE_QOS="${QUEUE_QOS:-n0064}"
MAX_JOBS="${MAX_JOBS:-8}"
POLL_SEC="${POLL_SEC:-60}"
CONFIG_DIR=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-dir) CONFIG_DIR="$2"; shift 2;;
    --out-root) OUT_ROOT="$2"; shift 2;;
    --max-jobs) MAX_JOBS="$2"; shift 2;;
    --poll-sec) POLL_SEC="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2;;
  esac
done

if [[ ! -f "$JOB_SCRIPT" ]]; then
  echo "Missing JOB_SCRIPT: $JOB_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$ANN_CKPT" ]]; then
  echo "Missing ANN_CKPT: $ANN_CKPT" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT" "$OUT_ROOT/slurm_queue"
QUEUE_DIR="$OUT_ROOT/slurm_queue"
LOG="$QUEUE_DIR/submitter.log"
QUEUE_TSV="$QUEUE_DIR/queue.tsv"
SUBMITTED_TSV="$QUEUE_DIR/submitted.tsv"
STATUS_TSV="$QUEUE_DIR/status.tsv"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"
}

if [[ -z "$CONFIG_DIR" ]]; then
  log "Generating configs under $OUT_ROOT"
  OUT_ROOT="$OUT_ROOT" "$REPO_ROOT/scripts/run_spikereg_paper_suite.sh" --mode configs | tee -a "$LOG"
  CONFIG_DIR="$OUT_ROOT/configs"
fi

if [[ ! -d "$CONFIG_DIR" ]]; then
  echo "Missing config dir: $CONFIG_DIR" >&2
  exit 1
fi

mapfile -t CONFIGS < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name '*.yaml' | sort)
if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
  echo "No YAML configs found in $CONFIG_DIR" >&2
  exit 1
fi

printf 'index\tname\tconfig\tstart_from_checkpoint\n' > "$QUEUE_TSV"
for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  name="$(basename "$cfg" .yaml)"
  start="$ANN_CKPT"
  if [[ "$name" == snn_scratch_* ]]; then
    start=""
  fi
  printf '%s\t%s\t%s\t%s\n' "$i" "$name" "$cfg" "$start" >> "$QUEUE_TSV"
done

printf 'job_id\tindex\tname\tconfig\tstart_from_checkpoint\trundir\tsubmitted_at\n' > "$SUBMITTED_TSV"
printf 'job_id\tindex\tname\tstate\trundir\tfinished_at\n' > "$STATUS_TSV"

declare -A ACTIVE_NAME=()
declare -A ACTIVE_INDEX=()
declare -A ACTIVE_RUNDIR=()
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

submit_one() {
  local idx="$1"
  local cfg="${CONFIGS[$idx]}"
  local name
  name="$(basename "$cfg" .yaml)"
  local start="$ANN_CKPT"
  if [[ "$name" == snn_scratch_* ]]; then
    start=""
  fi

  local sbatch_args=()
  if [[ -n "$SBATCH_ACCOUNT" ]]; then
    sbatch_args+=(-A "$SBATCH_ACCOUNT")
  fi
  if [[ -n "$SBATCH_PARTITION" ]]; then
    sbatch_args+=(-p "$SBATCH_PARTITION")
  fi
  if [[ -n "$QUEUE_QOS" ]]; then
    sbatch_args+=(--qos "$QUEUE_QOS")
  fi
  sbatch_args+=(
    --nodes "$QUEUE_NODES"
    --ntasks "$QUEUE_NTASKS"
    --ntasks-per-node "$QUEUE_NTASKS_PER_NODE"
    --gres "gpu:${QUEUE_GPUS_PER_NODE}"
    --cpus-per-task "$QUEUE_CPUS_PER_TASK"
    --mem "$QUEUE_MEM"
    --time "$QUEUE_TIME"
    -J "sr_${name:0:20}"
  )

  local export_vars="ALL,EXPERIMENT_ROOT=${EXPERIMENT_ROOT},SPIKEREG_QUEUE_NAME=${name},GPUS_PER_NODE=${QUEUE_GPUS_PER_NODE}"
  local cmd=(sbatch "${sbatch_args[@]}" --export="$export_vars" "$JOB_SCRIPT" --config "$cfg")
  if [[ -n "$start" ]]; then
    cmd+=(--start_from_checkpoint "$start")
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run submit [$idx/$(( ${#CONFIGS[@]} - 1 ))] $name: ${cmd[*]}"
    return
  fi

  log "submit [$idx/$(( ${#CONFIGS[@]} - 1 ))] $name"
  local out jid rundir
  out="$("${cmd[@]}")"
  printf '%s\n' "$out" | tee -a "$LOG"
  jid="$(printf '%s\n' "$out" | grep -o '[0-9]\+$' | tail -n1)"
  if [[ -z "$jid" ]]; then
    log "ERROR: could not parse sbatch job id for $name"
    exit 1
  fi
  rundir="$EXPERIMENT_ROOT/runs/$jid"
  ACTIVE_NAME["$jid"]="$name"
  ACTIVE_INDEX["$jid"]="$idx"
  ACTIVE_RUNDIR["$jid"]="$rundir"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$jid" "$idx" "$name" "$cfg" "$start" "$rundir" "$(date '+%F %T')" >> "$SUBMITTED_TSV"
}

active_count() {
  printf '%s\n' "${!ACTIVE_NAME[@]}" | sed '/^$/d' | wc -l
}

if [[ "$DRY_RUN" == "1" ]]; then
  log "Dry run: ${#CONFIGS[@]} configs, max jobs $MAX_JOBS"
  for i in "${!CONFIGS[@]}"; do
    submit_one "$i"
  done
  log "Dry run complete"
  exit 0
fi

log "Queue started: ${#CONFIGS[@]} configs, max jobs $MAX_JOBS, poll ${POLL_SEC}s"
log "Config dir: $CONFIG_DIR"
log "Submitted/status files: $SUBMITTED_TSV ; $STATUS_TSV"

while :; do
  for jid in "${!ACTIVE_NAME[@]}"; do
    state="$(sbatch_state "$jid")"
    if ! is_active_state "$state"; then
      name="${ACTIVE_NAME[$jid]}"
      idx="${ACTIVE_INDEX[$jid]}"
      rundir="${ACTIVE_RUNDIR[$jid]}"
      log "finished job=$jid name=$name state=$state"
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$jid" "$idx" "$name" "$state" "$rundir" "$(date '+%F %T')" >> "$STATUS_TSV"
      DONE["$jid"]="$state"
      unset 'ACTIVE_NAME[$jid]' 'ACTIVE_INDEX[$jid]' 'ACTIVE_RUNDIR[$jid]'
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

log "Queue complete: submitted=${#CONFIGS[@]} done=${#DONE[@]}"
log "Final status: $STATUS_TSV"
