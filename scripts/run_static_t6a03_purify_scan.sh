#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"

SRC_ROOT="${PROJECT_DIR}/src"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-${RUN_TS}_static_t6a03_purify_scan}"
RUN_ROOT="${PROJECT_DIR}/runs/${RUN_NAME}"
LOG_DIR="${RUN_ROOT}/train_logs"
CURVES_DIR="${RUN_ROOT}/curves"
SUMMARY_CSV="${RUN_ROOT}/summary.csv"
SUMMARY_JSON="${RUN_ROOT}/summary.json"
MANIFEST="${RUN_ROOT}/manifest.tsv"

DATASET="${DATASET:-Cifar10_dir0.5_bdoor0.2_nclient_100_badnet_adv10}"
ROUNDS="${ROUNDS:-600}"
EVAL_GAP="${EVAL_GAP:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BATCH_PARALLEL="${BATCH_PARALLEL:-0}"
ONLY_TAG="${ONLY_TAG:-}"

ensure_dir "${RUN_ROOT}"
ensure_dir "${LOG_DIR}"
ensure_dir "${CURVES_DIR}"

declare -a CONFIGS=()

add_config() {
  local tag="$1"
  local algorithm="$2"
  local purify_beta="$3"
  local purify_feature_beta="$4"
  local purify_logit_beta="$5"
  local purify_start_round="$6"
  local purify_layers="$7"
  local purify_teacher_momentum="$8"
  local purpose="$9"

  CONFIGS+=("${tag}|${algorithm}|${purify_beta}|${purify_feature_beta}|${purify_logit_beta}|${purify_start_round}|${purify_layers}|${purify_teacher_momentum}|${purpose}")
}

# Reference rerun. We already have HIT_CLCM_T6A03_BASE, but keeping one
# baseline in the same run prevents future confusion about code/data state.
add_config "S00_CLCM_T6A03_REF" "FedCLCM" 0.0 0.0 0.0 1 "layer4" 0.90 "same-run reference: tau6/alpha0.3 without purify"

# Important: purify_start_round is per-client selected-train count, not global
# round. With jr=0.1, start=10 roughly means the client has seen around 100
# global rounds; start=50 roughly means around 500 global rounds.
add_config "S01_ATT_L4_B0P003_S1" "FedCLCMPurify" 0.003 0.0 0.0 1 "layer4" 0.90 "very weak attention, active after teacher warmup"
add_config "S02_ATT_L4_B0P01_S1" "FedCLCMPurify" 0.01 0.0 0.0 1 "layer4" 0.90 "weak attention, early"
add_config "S03_ATT_L4_B0P03_S1" "FedCLCMPurify" 0.03 0.0 0.0 1 "layer4" 0.90 "previous beta but early instead of local-start50"
add_config "S04_ATT_L4_B0P003_S10" "FedCLCMPurify" 0.003 0.0 0.0 10 "layer4" 0.90 "very weak attention, moderate local delay"
add_config "S05_ATT_L4_B0P01_S10" "FedCLCMPurify" 0.01 0.0 0.0 10 "layer4" 0.90 "weak attention, moderate local delay"
add_config "S06_ATT_L4_B0P03_S10" "FedCLCMPurify" 0.03 0.0 0.0 10 "layer4" 0.90 "previous beta, moderate local delay"
add_config "S07_ATT_L4_B0P01_S20" "FedCLCMPurify" 0.01 0.0 0.0 20 "layer4" 0.90 "weak attention, later local delay"

# Layer placement test. If layer4 preserves semantic backdoor features, lower
# representation alignment should be less likely to raise ASR.
add_config "S08_ATT_L3_B0P01_S10" "FedCLCMPurify" 0.01 0.0 0.0 10 "layer3" 0.90 "lower semantic layer attention"
add_config "S09_ATT_L2L3_B0P01_S10" "FedCLCMPurify" 0.01 0.0 0.0 10 "layer2,layer3" 0.90 "mid-layer attention without layer4"
add_config "S10_ATT_L3_B0P03_S10" "FedCLCMPurify" 0.03 0.0 0.0 10 "layer3" 0.90 "stronger lower-layer attention"

# Teacher inertia test. If EMA is carrying poisoned global features, slower or
# faster teacher updates should visibly change ASR.
add_config "S11_ATT_L4_B0P01_S10_M0P5" "FedCLCMPurify" 0.01 0.0 0.0 10 "layer4" 0.50 "faster teacher adaptation"
add_config "S12_ATT_L4_B0P01_S10_M0P99" "FedCLCMPurify" 0.01 0.0 0.0 10 "layer4" 0.99 "slower teacher adaptation"

if [[ -n "${ONLY_TAG}" ]]; then
  FILTERED=()
  for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r tag _ <<< "${cfg}"
    if [[ "${tag}" == "${ONLY_TAG}" ]]; then
      FILTERED+=("${cfg}")
    fi
  done
  CONFIGS=("${FILTERED[@]}")
fi

if (( ${#CONFIGS[@]} == 0 )); then
  echo "[FATAL] no configs selected" >&2
  exit 1
fi

if [[ ! -f "${SRC_ROOT}/dataset/${DATASET}/config.json" ]]; then
  echo "[FATAL] missing dataset ${DATASET}. Generate it before running." >&2
  exit 1
fi

echo -e "tag\talgorithm\tdataset\tmodel\tnum_clients\tadv_clients\tjoin_ratio\tlocal_lr\tlr_head\tlocal_epochs\tplocal_epochs\trounds\teval_gap\trt_beta\tlambda_cl\taug_strength\tmask_tau\tmask_alpha\tpurify_beta\tpurify_feature_beta\tpurify_logit_beta\tpurify_start_round_local_selected_count\tpurify_layers\tpurify_teacher_momentum\tpurpose" > "${MANIFEST}"
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r tag algorithm purify_beta purify_feature_beta purify_logit_beta purify_start_round purify_layers purify_teacher_momentum purpose <<< "${cfg}"
  echo -e "${tag}\t${algorithm}\t${DATASET}\tResNet18\t100\t10\t0.1\t0.1\t0.1\t1\t1\t${ROUNDS}\t${EVAL_GAP}\t0.2\t0.2\t0.1\t6.0\t0.3\t${purify_beta}\t${purify_feature_beta}\t${purify_logit_beta}\t${purify_start_round}\t${purify_layers}\t${purify_teacher_momentum}\t${purpose}" >> "${MANIFEST}"
done

launch_one() {
  local cfg="$1"
  local gpu="$2"
  IFS='|' read -r tag algorithm purify_beta purify_feature_beta purify_logit_beta purify_start_round purify_layers purify_teacher_momentum purpose <<< "${cfg}"

  local log_file="${LOG_DIR}/${tag}.log"
  local -a cmd=(
    "${PYTHON_BIN}" -u main.py
    -dev cuda -did "${gpu}"
    -data "${DATASET}"
    -m ResNet18
    -algo "${algorithm}"
    -ncl 10 -nc 100 -jr 0.1 -lbs "${BATCH_SIZE}"
    -lr 0.1 -lr_head 0.1 -ls 1 -pls 1
    -gr "${ROUNDS}" -eg "${EVAL_GAP}"
    -go "${tag}"
    --num_adv_clients 10
    --rt_beta 0.2
    --lambda_cl 0.2
    --aug_strength 0.1
    --mask_tau 6.0
    --mask_alpha 0.3
    --enable_channel_mask true
    --adv_eps 0.0
    --adv_num_iter 0
  )

  if [[ "${algorithm}" == "FedCLCMPurify" ]]; then
    cmd+=(
      --purify_beta "${purify_beta}"
      --purify_feature_beta "${purify_feature_beta}"
      --purify_logit_beta "${purify_logit_beta}"
      --purify_start_round "${purify_start_round}"
      --purify_layers "${purify_layers}"
      --purify_teacher_momentum "${purify_teacher_momentum}"
      --purify_teacher_cpu_half true
    )
  fi

  echo "[RUN] ${tag} gpu=${gpu} algo=${algorithm} purpose=${purpose}"
  (
    cd "${SRC_ROOT}"
    "${cmd[@]}" > "${log_file}" 2>&1
  )
}

run_config_batches CONFIGS launch_one "${BATCH_PARALLEL}"
collect_metrics "${LOG_DIR}" "*.log" "${SUMMARY_CSV}" "${SUMMARY_JSON}" "${CURVES_DIR}"

echo "[READY] run_root=${RUN_ROOT}"
echo "[READY] manifest=${MANIFEST}"
echo "[READY] summary=${SUMMARY_CSV}"
