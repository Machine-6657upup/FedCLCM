#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

echo "[COMBO] start PFedBA FedCLCM sweep: ${BASE_TS}"
RUN_TS="${BASE_TS}_pfedba" RUN_NAME="${BASE_TS}_pfedba_clcm_sweep_4090" "${SCRIPT_DIR}/run_4090_pfedba_clcm_sweep_long.sh"

echo "[COMBO] start static/clean richness: ${BASE_TS}"
RUN_TS="${BASE_TS}_static" RUN_NAME="${BASE_TS}_thesis_richness_static_4090" "${SCRIPT_DIR}/run_4090_thesis_richness_static.sh"

echo "[COMBO] all done"
