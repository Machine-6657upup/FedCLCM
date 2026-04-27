#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
WATCH_NAME="${WATCH_NAME:-${RUN_TS}_watch_then_pfedba_clcmpur}"
WATCH_ROOT="${PROJECT_DIR}/runs/${WATCH_NAME}"
WATCH_LOG="${WATCH_ROOT}/watcher.out"
WATCH_REGEX="${WATCH_REGEX:-runs/20260427_hit_setting_.*queue_scripts}"
CHECK_INTERVAL="${CHECK_INTERVAL:-300}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-43200}"

mkdir -p "${WATCH_ROOT}"

log() {
  echo "[$(date '+%F %T')] $*"
}

current_jobs() {
  pgrep -af "${WATCH_REGEX}" || true
}

{
  log "watch_root=${WATCH_ROOT}"
  log "watch_regex=${WATCH_REGEX}"
  log "check_interval=${CHECK_INTERVAL}"
  log "max_wait_seconds=${MAX_WAIT_SECONDS}"
  log "next_mode=${MODE:-all}"
  log "next_gpus=${GPUS:-0 1 2 3}"
  log "next_batch_parallel=${BATCH_PARALLEL:-4}"
  log "next_run_name=${RUN_NAME:-${RUN_TS}_pfedba_clcmpur_after_static}"

  start_ts="$(date +%s)"
  while true; do
    jobs="$(current_jobs)"
    if [[ -z "${jobs}" ]]; then
      log "watched jobs are finished; launching PFedBA CLCMPur queue"
      break
    fi

    now_ts="$(date +%s)"
    elapsed=$((now_ts - start_ts))
    if (( elapsed > MAX_WAIT_SECONDS )); then
      log "timeout while waiting; aborting watcher without launching"
      exit 124
    fi

    log "still waiting (${elapsed}s elapsed)"
    echo "${jobs}"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
    sleep "${CHECK_INTERVAL}"
  done

  export MODE="${MODE:-all}"
  export GPUS="${GPUS:-0 1 2 3}"
  export BATCH_PARALLEL="${BATCH_PARALLEL:-4}"
  export RUN_NAME="${RUN_NAME:-${RUN_TS}_pfedba_clcmpur_after_static}"

  exec bash "${PROJECT_DIR}/scripts/run_pfedba_clcmpur_autorun.sh"
} >> "${WATCH_LOG}" 2>&1
