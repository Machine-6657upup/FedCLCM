#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/huangtu/PFL_clean_workspace/root_static/pfedba_local"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-${ROOT_DIR}/log/formal_defense_matrix_22_20260419_213225}"
LAUNCHER="${ROOT_DIR}/scripts/run_formal_defense_matrix_22_4gpu.sh"
FORCE_TAGS="${FORCE_TAGS:-E03,E07,E09}"
OUT_LOG="${ROOT_DIR}/log/formal_defense_matrix_22_remaining_screen_launcher_${RUN_TS}.out"

mkdir -p "${ROOT_DIR}/log" "${LOGDIR}"

env \
  RUN_TS="${RUN_TS}" \
  LOGDIR="${LOGDIR}" \
  FORCE_TAGS="${FORCE_TAGS}" \
  LAUNCH_MODE=screen \
  bash "${LAUNCHER}" > "${OUT_LOG}" 2>&1

echo "RUN_TS=${RUN_TS}"
echo "LOGDIR=${LOGDIR}"
echo "FORCE_TAGS=${FORCE_TAGS}"
echo "OUT_LOG=${OUT_LOG}"
