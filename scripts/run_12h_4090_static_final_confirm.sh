#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"

SRC_ROOT="${PROJECT_DIR}/src"
RUN_TS="${RUN_TS:-20260427_12h}"
RUN_NAME="${RUN_NAME:-${RUN_TS}_4090_static_final_confirm}"
RUN_ROOT="${PROJECT_DIR}/runs/${RUN_NAME}"
LOG_DIR="${RUN_ROOT}/train_logs"
CURVES_DIR="${RUN_ROOT}/curves"
SUMMARY_CSV="${RUN_ROOT}/summary.csv"
SUMMARY_JSON="${RUN_ROOT}/summary.json"
MANIFEST="${RUN_ROOT}/manifest.tsv"

GPU="${GPU:-0}"
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

echo -e "tag\tattack\tdataset\trounds\tlocal_epochs\tplocal_epochs\trt_beta\tlambda_cl\taug_strength\tmask_tau\tmask_alpha\tenable_channel_mask\tadv_eps\tadv_num_iter\tcosine_gate\tcosine_gate_threshold\tcosine_gate_alpha\tpurpose" > "${MANIFEST}"

declare -a CONFIGS=(
  "B04_600|badnet|${BADNET_DATASET}|${FINAL_ROUNDS}|0.20|0.20|0.10|6.0|0.30|true|0.00|0|0|0.30|0.50|final_confirm_low_asr"
  "B07_600|badnet|${BADNET_DATASET}|${FINAL_ROUNDS}|0.20|0.20|0.20|12.0|0.70|true|0.00|0|0|0.30|0.50|final_confirm_balanced"
  "BL02_600|blend|${BLEND_DATASET}|${FINAL_ROUNDS}|0.40|0.20|0.10|12.0|0.70|true|0.00|0|0|0.30|0.50|final_confirm_balanced"
  "BL06_600|blend|${BLEND_DATASET}|${FINAL_ROUNDS}|0.20|0.30|0.10|12.0|0.70|true|0.00|0|0|0.30|0.50|final_confirm_high_acc"
  "SG03_600|sig|${SIG_DATASET}|${FINAL_ROUNDS}|0.20|0.20|0.10|8.0|0.50|true|0.00|0|0|0.30|0.50|final_confirm_balanced"
  "SG04_600|sig|${SIG_DATASET}|${FINAL_ROUNDS}|0.20|0.20|0.10|6.0|0.30|true|0.00|0|0|0.30|0.50|final_confirm_low_asr"
)

run_one() {
  local cfg="$1"
  IFS='|' read -r tag attack dataset_name rounds rt_beta lambda_cl aug_strength mask_tau mask_alpha enable_channel_mask adv_eps adv_num_iter cosine_gate cosine_gate_threshold cosine_gate_alpha purpose <<< "${cfg}"

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
    cmd+=(--cosine_gate --cosine_gate_threshold "${cosine_gate_threshold}" --cosine_gate_alpha "${cosine_gate_alpha}")
  fi

  echo -e "${tag}\t${attack}\t${dataset_name}\t${rounds}\t${LOCAL_EPOCHS}\t${PLOCAL_EPOCHS}\t${rt_beta}\t${lambda_cl}\t${aug_strength}\t${mask_tau}\t${mask_alpha}\t${enable_channel_mask}\t${adv_eps}\t${adv_num_iter}\t${cosine_gate}\t${cosine_gate_threshold}\t${cosine_gate_alpha}\t${purpose}" >> "${MANIFEST}"
  echo "[RUN] ${tag} attack=${attack} gpu=${GPU} rounds=${rounds} purpose=${purpose}"
  (
    cd "${SRC_ROOT}"
    "${cmd[@]}" > "${log_file}" 2>&1
  )
}

for cfg in "${CONFIGS[@]}"; do
  run_one "${cfg}"
done

collect_metrics "${LOG_DIR}" "*.log" "${SUMMARY_CSV}" "${SUMMARY_JSON}" "${CURVES_DIR}"

echo "[READY] summary=${SUMMARY_CSV}"
