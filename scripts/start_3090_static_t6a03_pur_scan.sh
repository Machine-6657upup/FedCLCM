#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_NAME="${RUN_NAME:-20260427_static_t6a03_pur_scan_3090}"
RUN_ROOT="${PROJECT_DIR}/runs/${RUN_NAME}"
mkdir -p "${RUN_ROOT}"

export RUN_NAME
export GPUS="${GPUS:-0 1 2 3}"
export BATCH_PARALLEL="${BATCH_PARALLEL:-4}"
export ROUNDS="${ROUNDS:-600}"
export PYTHON_BIN="${PYTHON_BIN:-/home/huangtu/miniconda3/envs/torch/bin/python}"

nohup bash "${PROJECT_DIR}/scripts/run_static_t6a03_purify_scan.sh" \
  > "${RUN_ROOT}/launcher.out" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${RUN_ROOT}/launcher.pid"

echo "launcher_pid=${pid}"
echo "run_root=${RUN_ROOT}"
echo "launcher_log=${RUN_ROOT}/launcher.out"
