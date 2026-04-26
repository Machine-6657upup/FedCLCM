#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"

SRC_ROOT="${PROJECT_DIR}/src"
RUN_TS="${RUN_TS:-20260427_12h}"
RUN_NAME="${RUN_NAME:-${RUN_TS}_3090_localepoch_badnet_limit}"
RUN_ROOT="${PROJECT_DIR}/runs/${RUN_NAME}"
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
BATCH_SIZE="${BATCH_SIZE:-64}"
BASE_MODEL="${BASE_MODEL:-ResNet18}"
EVAL_GAP="${EVAL_GAP:-10}"
BATCH_PARALLEL="${BATCH_PARALLEL:-4}"

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

declare -a CONFIGS=()

add_config() {
  local tag="$1"
  local attack="$2"
  local dataset_name="$3"
  local rounds="$4"
  local local_epochs="$5"
  local plocal_epochs="$6"
  local rt_beta="$7"
  local lambda_cl="$8"
  local aug_strength="$9"
  local mask_tau="${10}"
  local mask_alpha="${11}"
  local enable_channel_mask="${12}"
  local adv_eps="${13}"
  local adv_num_iter="${14}"
  local cosine_gate="${15}"
  local cosine_gate_threshold="${16}"
  local cosine_gate_alpha="${17}"
  local purpose="${18}"

  CONFIGS+=("${tag}|${attack}|${dataset_name}|${rounds}|${local_epochs}|${plocal_epochs}|${rt_beta}|${lambda_cl}|${aug_strength}|${mask_tau}|${mask_alpha}|${enable_channel_mask}|${adv_eps}|${adv_num_iter}|${cosine_gate}|${cosine_gate_threshold}|${cosine_gate_alpha}|${purpose}")
}

add_epoch_sweep() {
  local attack="$1"
  local prefix="$2"
  local dataset_name="$3"
  local rt_beta="$4"
  local lambda_cl="$5"
  local aug_strength="$6"
  local mask_tau="$7"
  local mask_alpha="$8"

  # Compute-aware local epoch sweep for the 12h window.
  # E=10/20 are intentionally shorter; manifest records rounds for interpretation.
  add_config "${prefix}_LE01" "${attack}" "${dataset_name}" 300 1 1 "${rt_beta}" "${lambda_cl}" "${aug_strength}" "${mask_tau}" "${mask_alpha}" true 0.00 0 0 0.30 0.50 "local_epoch_sweep"
  add_config "${prefix}_LE02" "${attack}" "${dataset_name}" 200 2 2 "${rt_beta}" "${lambda_cl}" "${aug_strength}" "${mask_tau}" "${mask_alpha}" true 0.00 0 0 0.30 0.50 "local_epoch_sweep"
  add_config "${prefix}_LE05" "${attack}" "${dataset_name}" 120 5 5 "${rt_beta}" "${lambda_cl}" "${aug_strength}" "${mask_tau}" "${mask_alpha}" true 0.00 0 0 0.30 0.50 "local_epoch_sweep"
  add_config "${prefix}_LE10" "${attack}" "${dataset_name}" 80 10 10 "${rt_beta}" "${lambda_cl}" "${aug_strength}" "${mask_tau}" "${mask_alpha}" true 0.00 0 0 0.30 0.50 "local_epoch_sweep"
  add_config "${prefix}_LE20" "${attack}" "${dataset_name}" 50 20 20 "${rt_beta}" "${lambda_cl}" "${aug_strength}" "${mask_tau}" "${mask_alpha}" true 0.00 0 0 0.30 0.50 "local_epoch_sweep"
}

# Representative configs from completed 300-round sweeps.
add_epoch_sweep "badnet" "BN_B07" "${BADNET_DATASET}" 0.20 0.20 0.20 12.0 0.70
add_epoch_sweep "blend" "BL_BL06" "${BLEND_DATASET}" 0.20 0.30 0.10 12.0 0.70
add_epoch_sweep "sig" "SG_SG03" "${SIG_DATASET}" 0.20 0.20 0.10 8.0 0.50

# Focused badnet limit search around B04/B07. Goal: keep ACC near 0.75 while pushing ASR below B07.
add_config "BN_LIM01" "badnet" "${BADNET_DATASET}" 300 1 1 0.20 0.20 0.15 8.0 0.50 true 0.00 0 0 0.30 0.50 "badnet_limit"
add_config "BN_LIM02" "badnet" "${BADNET_DATASET}" 300 1 1 0.20 0.20 0.20 8.0 0.50 true 0.00 0 0 0.30 0.50 "badnet_limit"
add_config "BN_LIM03" "badnet" "${BADNET_DATASET}" 300 1 1 0.25 0.20 0.20 8.0 0.50 true 0.00 0 0 0.30 0.50 "badnet_limit"
add_config "BN_LIM04" "badnet" "${BADNET_DATASET}" 300 1 1 0.30 0.20 0.20 8.0 0.50 true 0.00 0 0 0.30 0.50 "badnet_limit"
add_config "BN_LIM05" "badnet" "${BADNET_DATASET}" 300 1 1 0.20 0.15 0.20 8.0 0.50 true 0.00 0 0 0.30 0.50 "badnet_limit"
add_config "BN_LIM06" "badnet" "${BADNET_DATASET}" 300 1 1 0.20 0.25 0.20 8.0 0.50 true 0.00 0 0 0.30 0.50 "badnet_limit"
add_config "BN_LIM07" "badnet" "${BADNET_DATASET}" 300 1 1 0.20 0.20 0.20 6.0 0.40 true 0.00 0 0 0.30 0.50 "badnet_limit"
add_config "BN_LIM08" "badnet" "${BADNET_DATASET}" 300 1 1 0.20 0.20 0.20 10.0 0.60 true 0.00 0 0 0.30 0.50 "badnet_limit"
add_config "BN_LIM09" "badnet" "${BADNET_DATASET}" 300 1 1 0.20 0.20 0.20 12.0 0.70 true 0.02 1 0 0.30 0.50 "badnet_limit_adv_light"
add_config "BN_LIM10" "badnet" "${BADNET_DATASET}" 300 1 1 0.20 0.20 0.15 8.0 0.50 true 0.02 1 0 0.30 0.50 "badnet_limit_adv_light"

launch_one() {
  local cfg="$1"
  local gpu="$2"
  IFS='|' read -r tag attack dataset_name rounds local_epochs plocal_epochs rt_beta lambda_cl aug_strength mask_tau mask_alpha enable_channel_mask adv_eps adv_num_iter cosine_gate cosine_gate_threshold cosine_gate_alpha purpose <<< "${cfg}"

  local log_file="${LOG_DIR}/${tag}.log"
  local -a cmd=(
    "${PYTHON_BIN}" -u main.py
    -dev cuda -did "${gpu}"
    -data "${dataset_name}"
    -m "${BASE_MODEL}" -algo FedCLCM
    -ncl 10 -nc "${NUM_CLIENTS}" -jr "${JOIN_RATIO}" -lbs "${BATCH_SIZE}"
    -lr "${LOCAL_LR}" -lr_head "${LR_HEAD}" -ls "${local_epochs}" -pls "${plocal_epochs}"
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

  echo -e "${tag}\t${attack}\t${dataset_name}\t${rounds}\t${local_epochs}\t${plocal_epochs}\t${rt_beta}\t${lambda_cl}\t${aug_strength}\t${mask_tau}\t${mask_alpha}\t${enable_channel_mask}\t${adv_eps}\t${adv_num_iter}\t${cosine_gate}\t${cosine_gate_threshold}\t${cosine_gate_alpha}\t${purpose}" >> "${MANIFEST}"
  echo "[RUN] ${tag} attack=${attack} gpu=${gpu} rounds=${rounds} le=${local_epochs} ple=${plocal_epochs} purpose=${purpose}"
  (
    cd "${SRC_ROOT}"
    "${cmd[@]}" > "${log_file}" 2>&1
  )
}

run_config_batches CONFIGS launch_one "${BATCH_PARALLEL}"
collect_metrics "${LOG_DIR}" "*.log" "${SUMMARY_CSV}" "${SUMMARY_JSON}" "${CURVES_DIR}"

echo "[READY] summary=${SUMMARY_CSV}"
