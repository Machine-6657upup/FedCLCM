#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"

SRC_ROOT="${PROJECT_DIR}/src"
DATASET_ROOT="${SRC_ROOT}/dataset"
RAWDATA_ROOT="${DATASET_ROOT}/rawdata"
GEN_DIR="${DATASET_ROOT}/utils"
WRAPPER="${PROJECT_DIR}/scripts/generate_dataset_via_existing.py"

RUN_TS="${RUN_TS:-$(timestamp)}"
RUN_ROOT="${PROJECT_DIR}/runs/${RUN_TS}_static_dataset_prep"
LOG_DIR="${RUN_ROOT}/dataset_logs"
MANIFEST="${RUN_ROOT}/manifest.tsv"

NUM_CLIENTS="${NUM_CLIENTS:-100}"
ADV_CLIENTS="${ADV_CLIENTS:-10}"
BACKDOOR_RATE="${BACKDOOR_RATE:-0.2}"
TARGET_LABEL="${TARGET_LABEL:-0}"
ALPHA="${ALPHA:-0.5}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
BATCH_SIZE="${BATCH_SIZE:-10}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
BLEND_ALPHA="${BLEND_ALPHA:-0.2}"
SIG_DELTA="${SIG_DELTA:-0.11764705882352941}"
SIG_F="${SIG_F:-6}"
SIG_LABEL_MODE="${SIG_LABEL_MODE:-dirty}"

BADNET_DATASET="Cifar10_dir${ALPHA}_bdoor${BACKDOOR_RATE}_nclient_${NUM_CLIENTS}_badnet_adv${ADV_CLIENTS}"
BLEND_DATASET="Cifar10_dir${ALPHA}_bdoor${BACKDOOR_RATE}_nclient_${NUM_CLIENTS}_blend_adv${ADV_CLIENTS}"
SIG_DATASET="Cifar10_dir${ALPHA}_bdoor${BACKDOOR_RATE}_nclient_${NUM_CLIENTS}_sig_adv${ADV_CLIENTS}"

ensure_dir "${DATASET_ROOT}"
ensure_dir "${RAWDATA_ROOT}"
ensure_dir "${LOG_DIR}"

if [[ ! -d "${GEN_DIR}" ]]; then
  echo "[FATAL] missing generator directory: ${GEN_DIR}" >&2
  exit 1
fi

echo -e "dataset_name\tgenerator\tstatus\tlog_file" > "${MANIFEST}"

run_dataset() {
  local dataset_name="$1"
  local generator_name="$2"
  local log_name="$3"
  shift 3

  local config_path="${DATASET_ROOT}/${dataset_name}/config.json"
  local log_file="${LOG_DIR}/${log_name}"
  local -a cmd=(
    "${PYTHON_BIN}" "${WRAPPER}"
    --generator "${GEN_DIR}/${generator_name}"
    --dir-path "${DATASET_ROOT}/${dataset_name}"
    --rawdata-path "${RAWDATA_ROOT}"
    --num-clients "${NUM_CLIENTS}"
    --backdoor-rate "${BACKDOOR_RATE}"
    --adversary-num "${ADV_CLIENTS}"
    --target-y "${TARGET_LABEL}"
    --alpha "${ALPHA}"
    --train-ratio "${TRAIN_RATIO}"
    --batch-size "${BATCH_SIZE}"
    --partition dir
    --niid
    --balance
  )

  echo "[DATASET] ensuring ${dataset_name}"
  cmd+=( "$@" )
  if [[ "${FORCE_REBUILD}" == "1" ]]; then
    cmd+=( --force-rebuild )
  fi
  "${cmd[@]}" > "${log_file}" 2>&1

  echo -e "${dataset_name}\t${generator_name}\tgenerated\t${log_file}" >> "${MANIFEST}"
}

run_dataset "${BADNET_DATASET}" "generate_Cifar10_badnet.py" "badnet.log"
run_dataset "${BLEND_DATASET}" "generate_Cifar10_blend.py" "blend.log" \
  --blend-alpha "${BLEND_ALPHA}"
run_dataset "${SIG_DATASET}" "generate_Cifar10_sig.py" "sig.log" \
  --sig-delta "${SIG_DELTA}" \
  --sig-f "${SIG_F}" \
  --sig-label-mode "${SIG_LABEL_MODE}"

echo "[READY] dataset manifest: ${MANIFEST}"
