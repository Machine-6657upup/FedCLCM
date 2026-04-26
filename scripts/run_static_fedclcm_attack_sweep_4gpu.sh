#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"

SRC_ROOT="${PROJECT_DIR}/src"
RUN_TS="${RUN_TS:-$(timestamp)}"
RUN_NAME="${RUN_NAME:-${RUN_TS}_static_fedclcm_attack_sweep_3090}"
RUN_ROOT="${PROJECT_DIR}/runs/${RUN_NAME}"
LOG_DIR="${RUN_ROOT}/train_logs"
CURVES_DIR="${RUN_ROOT}/curves"
SUMMARY_CSV="${RUN_ROOT}/summary.csv"
SUMMARY_JSON="${RUN_ROOT}/summary.json"
MANIFEST="${RUN_ROOT}/manifest.tsv"

ATTACKS="${ATTACKS:-blend sig}"
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
GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-300}"
BATCH_PARALLEL="${BATCH_PARALLEL:-4}"

ensure_dir "${RUN_ROOT}"
ensure_dir "${LOG_DIR}"
ensure_dir "${CURVES_DIR}"

dataset_for_attack() {
  local attack="$1"
  case "${attack}" in
    badnet|blend|sig)
      printf 'Cifar10_dir0.5_bdoor%s_nclient_%s_%s_adv%s' \
        "${BACKDOOR_RATE}" "${NUM_CLIENTS}" "${attack}" "${ADV_CLIENTS}"
      ;;
    *)
      echo "[FATAL] unsupported attack=${attack}" >&2
      exit 1
      ;;
  esac
}

for attack in ${ATTACKS}; do
  dataset_name="$(dataset_for_attack "${attack}")"
  if [[ ! -f "${SRC_ROOT}/dataset/${dataset_name}/config.json" ]]; then
    echo "[FATAL] missing dataset ${dataset_name}" >&2
    exit 1
  fi
done

echo -e "tag\tattack\tdataset\trounds\trt_beta\tlambda_cl\taug_strength\tmask_tau\tmask_alpha\tenable_channel_mask\tadv_eps\tadv_num_iter\tcosine_gate\tcosine_gate_threshold\tcosine_gate_alpha" > "${MANIFEST}"

declare -a CONFIGS=()

add_attack_configs() {
  local attack="$1"
  local prefix="$2"
  local dataset_name
  dataset_name="$(dataset_for_attack "${attack}")"

  CONFIGS+=(
    "${prefix}00|${attack}|${dataset_name}|${GLOBAL_ROUNDS}|0.20|0.20|0.10|12.0|0.70|true|0.00|0|0|0.30|0.50"
    "${prefix}01|${attack}|${dataset_name}|${GLOBAL_ROUNDS}|0.30|0.20|0.10|12.0|0.70|true|0.00|0|0|0.30|0.50"
    "${prefix}02|${attack}|${dataset_name}|${GLOBAL_ROUNDS}|0.40|0.20|0.10|12.0|0.70|true|0.00|0|0|0.30|0.50"
    "${prefix}03|${attack}|${dataset_name}|${GLOBAL_ROUNDS}|0.20|0.20|0.10|8.0|0.50|true|0.00|0|0|0.30|0.50"
    "${prefix}04|${attack}|${dataset_name}|${GLOBAL_ROUNDS}|0.20|0.20|0.10|6.0|0.30|true|0.00|0|0|0.30|0.50"
    "${prefix}05|${attack}|${dataset_name}|${GLOBAL_ROUNDS}|0.20|0.10|0.10|12.0|0.70|true|0.00|0|0|0.30|0.50"
    "${prefix}06|${attack}|${dataset_name}|${GLOBAL_ROUNDS}|0.20|0.30|0.10|12.0|0.70|true|0.00|0|0|0.30|0.50"
    "${prefix}07|${attack}|${dataset_name}|${GLOBAL_ROUNDS}|0.20|0.20|0.20|12.0|0.70|true|0.00|0|0|0.30|0.50"
    "${prefix}08|${attack}|${dataset_name}|${GLOBAL_ROUNDS}|0.20|0.20|0.10|12.0|0.70|true|0.05|3|0|0.30|0.50"
    "${prefix}09|${attack}|${dataset_name}|${GLOBAL_ROUNDS}|0.20|0.20|0.10|12.0|0.70|true|0.00|0|1|0.30|0.50"
    "${prefix}10|${attack}|${dataset_name}|${GLOBAL_ROUNDS}|0.20|0.20|0.10|12.0|0.70|false|0.00|0|0|0.30|0.50"
  )
}

for attack in ${ATTACKS}; do
  case "${attack}" in
    badnet) add_attack_configs "${attack}" "BN" ;;
    blend) add_attack_configs "${attack}" "BL" ;;
    sig) add_attack_configs "${attack}" "SG" ;;
  esac
done

launch_one() {
  local cfg="$1"
  local gpu="$2"
  IFS='|' read -r tag attack dataset_name rounds rt_beta lambda_cl aug_strength mask_tau mask_alpha enable_channel_mask adv_eps adv_num_iter cosine_gate cosine_gate_threshold cosine_gate_alpha <<< "${cfg}"

  local log_file="${LOG_DIR}/${tag}.log"
  local -a cmd=(
    "${PYTHON_BIN}" -u main.py
    -dev cuda -did "${gpu}"
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

  echo -e "${tag}\t${attack}\t${dataset_name}\t${rounds}\t${rt_beta}\t${lambda_cl}\t${aug_strength}\t${mask_tau}\t${mask_alpha}\t${enable_channel_mask}\t${adv_eps}\t${adv_num_iter}\t${cosine_gate}\t${cosine_gate_threshold}\t${cosine_gate_alpha}" >> "${MANIFEST}"
  echo "[RUN] ${tag} attack=${attack} gpu=${gpu}"
  (
    cd "${SRC_ROOT}"
    "${cmd[@]}" > "${log_file}" 2>&1
  )
}

run_config_batches CONFIGS launch_one "${BATCH_PARALLEL}"
collect_metrics "${LOG_DIR}" "*.log" "${SUMMARY_CSV}" "${SUMMARY_JSON}" "${CURVES_DIR}"

echo "[READY] summary=${SUMMARY_CSV}"
