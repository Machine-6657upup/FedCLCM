#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"

SRC_ROOT="${PROJECT_DIR}/src"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-${RUN_TS}_static_clcm_taualpha_scan}"
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
  local mask_tau="$2"
  local mask_alpha="$3"
  local purpose="$4"
  CONFIGS+=("${tag}|${mask_tau}|${mask_alpha}|${purpose}")
}

# Known good point from B04/HIT_CLCM_T6A03_BASE.
add_config "C00_T6_A0P30_REF" 6.0 0.30 "reference: current best static BadNet CLCM"

# Tau neighborhood at fixed alpha=0.30.
add_config "C01_T5_A0P30" 5.0 0.30 "stronger threshold, same downweight"
add_config "C02_T4_A0P30" 4.0 0.30 "much stronger threshold, same downweight"
add_config "C03_T7_A0P30" 7.0 0.30 "slightly milder threshold, same downweight"
add_config "C04_T8_A0P30" 8.0 0.30 "milder threshold, same downweight"

# Alpha neighborhood at fixed tau=6.
add_config "C05_T6_A0P20" 6.0 0.20 "same threshold, stronger downweight"
add_config "C06_T6_A0P10" 6.0 0.10 "same threshold, very strong downweight"
add_config "C07_T6_A0P40" 6.0 0.40 "same threshold, milder downweight"
add_config "C08_T6_A0P50" 6.0 0.50 "same threshold, much milder downweight"

# Joint stronger settings: likely lower ASR, may hurt ACC.
add_config "C09_T5_A0P20" 5.0 0.20 "joint stronger mask"
add_config "C10_T5_A0P10" 5.0 0.10 "joint very strong mask"
add_config "C11_T4_A0P20" 4.0 0.20 "aggressive threshold plus strong downweight"

# Joint milder settings: test whether ACC can improve while keeping ASR low.
add_config "C12_T7_A0P20" 7.0 0.20 "milder threshold but stronger downweight"
add_config "C13_T7_A0P40" 7.0 0.40 "milder threshold and milder downweight"

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

echo -e "tag\talgorithm\tdataset\tmodel\tnum_clients\tadv_clients\tjoin_ratio\tlocal_lr\tlr_head\tlocal_epochs\tplocal_epochs\trounds\teval_gap\trt_beta\tlambda_cl\taug_strength\tmask_tau\tmask_alpha\tpurpose" > "${MANIFEST}"
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r tag mask_tau mask_alpha purpose <<< "${cfg}"
  echo -e "${tag}\tFedCLCM\t${DATASET}\tResNet18\t100\t10\t0.1\t0.1\t0.1\t1\t1\t${ROUNDS}\t${EVAL_GAP}\t0.2\t0.2\t0.1\t${mask_tau}\t${mask_alpha}\t${purpose}" >> "${MANIFEST}"
done

launch_one() {
  local cfg="$1"
  local gpu="$2"
  IFS='|' read -r tag mask_tau mask_alpha purpose <<< "${cfg}"

  local log_file="${LOG_DIR}/${tag}.log"
  local -a cmd=(
    "${PYTHON_BIN}" -u main.py
    -dev cuda -did "${gpu}"
    -data "${DATASET}"
    -m ResNet18
    -algo FedCLCM
    -ncl 10 -nc 100 -jr 0.1 -lbs "${BATCH_SIZE}"
    -lr 0.1 -lr_head 0.1 -ls 1 -pls 1
    -gr "${ROUNDS}" -eg "${EVAL_GAP}"
    -go "${tag}"
    --num_adv_clients 10
    --rt_beta 0.2
    --lambda_cl 0.2
    --aug_strength 0.1
    --mask_tau "${mask_tau}"
    --mask_alpha "${mask_alpha}"
    --enable_channel_mask true
    --adv_eps 0.0
    --adv_num_iter 0
  )

  echo "[RUN] ${tag} gpu=${gpu} tau=${mask_tau} alpha=${mask_alpha} purpose=${purpose}"
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
