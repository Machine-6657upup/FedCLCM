#!/usr/bin/env bash
set -euo pipefail

LOG="${LOG:-/home/huangtu/FedCLCM_purify/runs/launchers/skip_L06_BLEND_LE5_watcher.out}"
PATTERN="${PATTERN:-python.*main.py.*-go L06_BLEND_LE5}"
MAX_STEPS="${MAX_STEPS:-7200}"
SLEEP_SECONDS="${SLEEP_SECONDS:-5}"

echo "[START] skip watcher $(date)" >> "${LOG}"

for _ in $(seq 1 "${MAX_STEPS}"); do
  pids="$(pgrep -f "${PATTERN}" || true)"
  if [[ -n "${pids}" ]]; then
    echo "[KILL] $(date) ${pids}" >> "${LOG}"
    # Intentionally kill only the exact L06 process; other experiments are left intact.
    kill ${pids}
    exit 0
  fi
  sleep "${SLEEP_SECONDS}"
done

echo "[END] no L06 found $(date)" >> "${LOG}"
