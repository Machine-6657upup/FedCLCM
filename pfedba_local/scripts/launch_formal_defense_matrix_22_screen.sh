#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/huangtu/PFL_clean_workspace/root_static/pfedba_local"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LAUNCHER="${ROOT_DIR}/scripts/run_formal_defense_matrix_22_4gpu.sh"
OUT_LOG="${ROOT_DIR}/log/formal_defense_matrix_22_screen_launcher_${RUN_TS}.out"

mkdir -p "${ROOT_DIR}/log"

env RUN_TS="${RUN_TS}" LAUNCH_MODE=screen bash "${LAUNCHER}" > "${OUT_LOG}" 2>&1

echo "RUN_TS=${RUN_TS}"
echo "OUT_LOG=${OUT_LOG}"
echo "LOGDIR=${ROOT_DIR}/log/formal_defense_matrix_22_${RUN_TS}"
