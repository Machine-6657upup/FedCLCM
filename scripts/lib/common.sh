#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

timestamp() {
  date +%Y%m%d_%H%M%S
}

ensure_dir() {
  mkdir -p "$1"
}

resolve_gpus() {
  local gpus_str="${GPUS:-0 1 2 3}"
  local -a gpus=()
  IFS=' ' read -r -a gpus <<< "${gpus_str}"
  printf '%s\n' "${gpus[@]}"
}

load_gpu_array() {
  mapfile -t GPU_LIST < <(resolve_gpus)
  if (( ${#GPU_LIST[@]} == 0 )); then
    echo "ERROR: no GPUs resolved from GPUS environment variable." >&2
    exit 1
  fi
}

run_config_batches() {
  local -n configs_ref="$1"
  local launcher_fn="$2"
  local batch_size="${3:-0}"

  load_gpu_array

  if (( batch_size <= 0 || batch_size > ${#GPU_LIST[@]} )); then
    batch_size="${#GPU_LIST[@]}"
  fi

  local idx=0
  while (( idx < ${#configs_ref[@]} )); do
    local -a pids=()
    local -a labels=()
    local slot=0

    while (( slot < batch_size && idx < ${#configs_ref[@]} )); do
      local cfg="${configs_ref[$idx]}"
      local gpu="${GPU_LIST[$slot]}"
      local label
      label="$(printf '%s' "${cfg}" | awk -F'|' '{print $1}')"
      labels+=("${label}")
      "${launcher_fn}" "${cfg}" "${gpu}" &
      pids+=("$!")
      idx=$((idx + 1))
      slot=$((slot + 1))
    done

    echo "Batch running: ${labels[*]}"
    wait "${pids[@]}"
    echo "Batch finished: ${labels[*]}"
  done
}

collect_metrics() {
  local log_dir="$1"
  local pattern="$2"
  local summary_csv="$3"
  local summary_json="$4"
  local curves_dir="$5"

  ensure_dir "$(dirname "${summary_csv}")"
  ensure_dir "$(dirname "${summary_json}")"
  ensure_dir "${curves_dir}"

  "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/collect_metrics.py" \
    --log-dir "${log_dir}" \
    --pattern "${pattern}" \
    --summary-csv "${summary_csv}" \
    --summary-json "${summary_json}" \
    --curves-dir "${curves_dir}"
}
