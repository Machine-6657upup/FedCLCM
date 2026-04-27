#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

WATCH_NAME="${WATCH_NAME:-20260427_watch_then_pfedba_clcmpur_3090}"
RUN_ROOT="${PROJECT_DIR}/runs/${WATCH_NAME}"
mkdir -p "${RUN_ROOT}"

export WATCH_NAME
export RUN_NAME="${RUN_NAME:-20260427_pfedba_clcmpur_after_static_3090}"
export MODE="${MODE:-all}"
export GPUS="${GPUS:-0 1 2 3}"
export BATCH_PARALLEL="${BATCH_PARALLEL:-4}"
export ROUNDS="${ROUNDS:-1000}"
export PYTHON_BIN="${PYTHON_BIN:-/home/huangtu/miniconda3/envs/torch/bin/python}"
export CHECK_INTERVAL="${CHECK_INTERVAL:-300}"
export MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-43200}"
export WATCH_REGEX="${WATCH_REGEX:-runs/20260427_hit_setting_.*queue_scripts}"

nohup bash "${PROJECT_DIR}/scripts/watch_then_run_pfedba_clcmpur.sh" \
  > "${RUN_ROOT}/starter.out" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${RUN_ROOT}/watcher.pid"

echo "watcher_pid=${pid}"
echo "watch_root=${RUN_ROOT}"
echo "watch_log=${RUN_ROOT}/watcher.out"
echo "next_run=${PROJECT_DIR}/runs/${RUN_NAME}"
