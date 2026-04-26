#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"

SRC_ROOT="${PROJECT_DIR}/src"
RUN_TS="${RUN_TS:-$(timestamp)}"
MODE="${MODE:-badnet_scan}"
GPU="${GPU:-0}"
RUN_ROOT="${PROJECT_DIR}/runs/${RUN_TS}_static_fedclcm_${MODE}"
LOG_DIR="${RUN_ROOT}/train_logs"
CURVES_DIR="${RUN_ROOT}/curves"
SUMMARY_CSV="${RUN_ROOT}/summary.csv"
SUMMARY_JSON="${RUN_ROOT}/summary.json"
MANIFEST="${RUN_ROOT}/manifest.tsv"

NUM_CLIENTS="${NUM_CLIENTS:-100}"
ADV_CLIENTS="${ADV_CLIENTS:-10}"
BACKDOOR_RATE="${BACKDOOR_RATE:-0.2}"
JOIN_RATIO="${JOIN_RATIO:-0.1}"
LOCAL_LR="${LOCAL_LR:-0.1}"
LR_HEAD="${LR_HEAD:-0.1}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
PLOCAL_EPOCHS="${PLOCAL_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BASE_MODEL="${BASE_MODEL:-ResNet18}"
EVAL_GAP="${EVAL_GAP:-10}"
SCAN_ROUNDS="${SCAN_ROUNDS:-300}"
FINAL_ROUNDS="${FINAL_ROUNDS:-600}"

BADNET_DATASET="Cifar10_dir0.5_bdoor${BACKDOOR_RATE}_nclient_${NUM_CLIENTS}_badnet_adv${ADV_CLIENTS}"
BLEND_DATASET="Cifar10_dir0.5_bdoor${BACKDOOR_RATE}_nclient_${NUM_CLIENTS}_blend_adv${ADV_CLIENTS}"
SIG_DATASET="Cifar10_dir0.5_bdoor${BACKDOOR_RATE}_nclient_${NUM_CLIENTS}_sig_adv${ADV_CLIENTS}"

ensure_dir "${RUN_ROOT}"
ensure_dir "${LOG_DIR}"
ensure_dir "${CURVES_DIR}"

for dataset_name in "${BADNET_DATASET}" "${BLEND_DATASET}" "${SIG_DATASET}"; do
  if [[ ! -f "${SRC_ROOT}/dataset/${dataset_name}/config.json" ]]; then
    echo "[FATAL] missing dataset ${dataset_name}. Run scripts/prepare_static_datasets_srcutils.sh first." >&2
    exit 1
  fi
done

echo -e "tag\tdataset\trounds\trt_beta\tlambda_cl\taug_strength\tmask_tau\tmask_alpha\tenable_channel_mask\tadv_eps\tadv_num_iter\tcosine_gate\tcosine_gate_threshold\tcosine_gate_alpha" > "${MANIFEST}"

run_one() {
  local tag="$1"
  local dataset_name="$2"
  local rounds="$3"
  local rt_beta="$4"
  local lambda_cl="$5"
  local aug_strength="$6"
  local mask_tau="$7"
  local mask_alpha="$8"
  local enable_channel_mask="$9"
  local adv_eps="${10}"
  local adv_num_iter="${11}"
  local cosine_gate="${12}"
  local cosine_gate_threshold="${13}"
  local cosine_gate_alpha="${14}"

  local log_file="${LOG_DIR}/${tag}.log"
  local -a cmd=(
    "${PYTHON_BIN}" -u main.py
    -dev cuda -did "${GPU}"
    -data "${dataset_name}"
    -m "${BASE_MODEL}" -algo FedCLCM
    -ncl 10 -nc "${NUM_CLIENTS}" -jr "${JOIN_RATIO}" -lbs "${BATCH_SIZE}"
    -lr "${LOCAL_LR}" -lr_head "${LR_HEAD}" -ls "${LOCAL_EPOCHS}" -pls "${PLOCAL_EPOCHS}"
    -gr "${rounds}" -eg "${EVAL_GAP}"
    -go "${tag}"
    --num_adv_clients "${ADV_CLIENTS}"
    --rt_beta "${rt_beta}"
    --lambda_cl "${lambda_cl}"
    --aug_strength "${aug_strength}"
    --mask_tau "${mask_tau}"
    --mask_alpha "${mask_alpha}"
    --enable_channel_mask "${enable_channel_mask}"
    --adv_eps "${adv_eps}"
    --adv_num_iter "${adv_num_iter}"
  )

  if [[ "${cosine_gate}" == "1" ]]; then
    cmd+=(
      --cosine_gate
      --cosine_gate_threshold "${cosine_gate_threshold}"
      --cosine_gate_alpha "${cosine_gate_alpha}"
    )
  fi

  echo -e "${tag}\t${dataset_name}\t${rounds}\t${rt_beta}\t${lambda_cl}\t${aug_strength}\t${mask_tau}\t${mask_alpha}\t${enable_channel_mask}\t${adv_eps}\t${adv_num_iter}\t${cosine_gate}\t${cosine_gate_threshold}\t${cosine_gate_alpha}" >> "${MANIFEST}"
  echo "[RUN] ${tag}"
  (
    cd "${SRC_ROOT}"
    "${cmd[@]}" > "${log_file}" 2>&1
  )
}

declare -a CONFIGS=()

case "${MODE}" in
  badnet_scan)
    CONFIGS=(
      "B00|${BADNET_DATASET}|${SCAN_ROUNDS}|0.20|0.20|0.10|12.0|0.70|true|0.00|0|0|0.30|0.50"
      "B01|${BADNET_DATASET}|${SCAN_ROUNDS}|0.30|0.20|0.10|12.0|0.70|true|0.00|0|0|0.30|0.50"
      "B02|${BADNET_DATASET}|${SCAN_ROUNDS}|0.40|0.20|0.10|12.0|0.70|true|0.00|0|0|0.30|0.50"
      "B03|${BADNET_DATASET}|${SCAN_ROUNDS}|0.20|0.20|0.10|8.0|0.50|true|0.00|0|0|0.30|0.50"
      "B04|${BADNET_DATASET}|${SCAN_ROUNDS}|0.20|0.20|0.10|6.0|0.30|true|0.00|0|0|0.30|0.50"
      "B05|${BADNET_DATASET}|${SCAN_ROUNDS}|0.20|0.10|0.10|12.0|0.70|true|0.00|0|0|0.30|0.50"
      "B06|${BADNET_DATASET}|${SCAN_ROUNDS}|0.20|0.30|0.10|12.0|0.70|true|0.00|0|0|0.30|0.50"
      "B07|${BADNET_DATASET}|${SCAN_ROUNDS}|0.20|0.20|0.20|12.0|0.70|true|0.00|0|0|0.30|0.50"
      "B08|${BADNET_DATASET}|${SCAN_ROUNDS}|0.20|0.20|0.10|12.0|0.70|true|0.05|3|0|0.30|0.50"
      "B09|${BADNET_DATASET}|${SCAN_ROUNDS}|0.20|0.20|0.10|12.0|0.70|true|0.00|0|1|0.30|0.50"
      "B10|${BADNET_DATASET}|${SCAN_ROUNDS}|0.20|0.20|0.10|12.0|0.70|false|0.00|0|0|0.30|0.50"
    )
    ;;
  cross_attack_baseline)
    CONFIGS=(
      "X01_badnet|${BADNET_DATASET}|${FINAL_ROUNDS}|0.20|0.20|0.10|12.0|0.70|true|0.00|0|0|0.30|0.50"
      "X02_blend|${BLEND_DATASET}|${FINAL_ROUNDS}|0.20|0.20|0.10|12.0|0.70|true|0.00|0|0|0.30|0.50"
      "X03_sig|${SIG_DATASET}|${FINAL_ROUNDS}|0.20|0.20|0.10|12.0|0.70|true|0.00|0|0|0.30|0.50"
    )
    ;;
  *)
    echo "[FATAL] unsupported MODE=${MODE}" >&2
    exit 1
    ;;
esac

for config in "${CONFIGS[@]}"; do
  IFS='|' read -r tag dataset_name rounds rt_beta lambda_cl aug_strength mask_tau mask_alpha enable_channel_mask adv_eps adv_num_iter cosine_gate cosine_gate_threshold cosine_gate_alpha <<< "${config}"
  run_one \
    "${tag}" \
    "${dataset_name}" \
    "${rounds}" \
    "${rt_beta}" \
    "${lambda_cl}" \
    "${aug_strength}" \
    "${mask_tau}" \
    "${mask_alpha}" \
    "${enable_channel_mask}" \
    "${adv_eps}" \
    "${adv_num_iter}" \
    "${cosine_gate}" \
    "${cosine_gate_threshold}" \
    "${cosine_gate_alpha}"
done

collect_metrics "${LOG_DIR}" "*.log" "${SUMMARY_CSV}" "${SUMMARY_JSON}" "${CURVES_DIR}"
echo "[READY] summary=${SUMMARY_CSV}"
