#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"

PFEDBA_ROOT="${PROJECT_DIR}/pfedba_local"
PFEDBA_MAIN="${PFEDBA_ROOT}/main.py"
PYTHON_BIN="${PYTHON_BIN:-/home/fch/miniconda3/envs/fedclcm-static/bin/python}"
GPU_ID="${GPU_ID:-0}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-${RUN_TS}_next_pfedba_refine_4090}"
RUN_ROOT="${PROJECT_DIR}/runs/${RUN_NAME}"
LOG_DIR="${RUN_ROOT}/train_logs"
STATUS_DIR="${RUN_ROOT}/status"
MANIFEST="${RUN_ROOT}/manifest.tsv"
ROUNDS="${ROUNDS:-1000}"
EVAL_GAP="${EVAL_GAP:-10}"
ONLY_TAG="${ONLY_TAG:-}"

mkdir -p "${LOG_DIR}" "${STATUS_DIR}" "${PFEDBA_ROOT}/results"

declare -a CONFIGS=()

add_config() {
  local tag="$1"
  local lr="$2"
  local mask_tau="$3"
  local mask_alpha="$4"
  local plocal_epochs="$5"
  local adv_eps="$6"
  local adv_num_iter="$7"
  local purpose="$8"
  CONFIGS+=("${tag}|${lr}|${mask_tau}|${mask_alpha}|${plocal_epochs}|${adv_eps}|${adv_num_iter}|${purpose}")
}

add_config "PFR_LR004_T50A025" 0.04 5.0 0.25 1 0.00 0 "refine_lr004_tau5_alpha025"
add_config "PFR_LR004_T50A030" 0.04 5.0 0.30 1 0.00 0 "refine_lr004_tau5_alpha030"
add_config "PFR_LR005_T45A025" 0.05 4.5 0.25 1 0.00 0 "refine_lr005_tau45_alpha025"
add_config "PFR_LR005_T45A030" 0.05 4.5 0.30 1 0.00 0 "refine_lr005_tau45_alpha030"
add_config "PFR_LR005_T50A025" 0.05 5.0 0.25 1 0.00 0 "refine_lr005_tau5_alpha025"
add_config "PFR_LR005_T55A025" 0.05 5.5 0.25 1 0.00 0 "refine_lr005_tau55_alpha025"
add_config "PFR_LR006_T50A025" 0.06 5.0 0.25 1 0.00 0 "refine_lr006_tau5_alpha025"
add_config "PFR_LR006_T50A030" 0.06 5.0 0.30 1 0.00 0 "refine_lr006_tau5_alpha030"
add_config "PFR_LR005_T50A030_PLE3" 0.05 5.0 0.30 3 0.00 0 "ple3_lr005_tau5_alpha030"
add_config "PFR_LR005_T50A030_PLE5" 0.05 5.0 0.30 5 0.00 0 "ple5_lr005_tau5_alpha030"
add_config "PFR_LR005_T50A030_PGD" 0.05 5.0 0.30 1 0.02 3 "pgd_light_lr005_tau5_alpha030"
add_config "PFR_LR008_T80A050_GUARD" 0.08 8.0 0.50 1 0.00 0 "high_acc_guard_lr008_tau8_alpha05"

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

if [[ ! -f "${PFEDBA_MAIN}" ]]; then
  echo "[FATAL] missing ${PFEDBA_MAIN}" >&2
  exit 1
fi

echo -e "tag\tlr\tlr_head\tlambda_cl\tmask_tau\tmask_alpha\tlocal_epochs\tplocal_epochs\tadv_eps\tadv_num_iter\trounds\tentry\tpurpose" > "${MANIFEST}"
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r tag lr tau alpha plocal_epochs adv_eps adv_num_iter purpose <<< "${cfg}"
  echo -e "${tag}\t${lr}\t${lr}\t0.05\t${tau}\t${alpha}\t1\t${plocal_epochs}\t${adv_eps}\t${adv_num_iter}\t${ROUNDS}\tpfedba_local/main.py\t${purpose}" >> "${MANIFEST}"
done

run_one() {
  local cfg="$1"
  IFS='|' read -r tag lr mask_tau mask_alpha plocal_epochs adv_eps adv_num_iter purpose <<< "${cfg}"
  local log_file="${LOG_DIR}/${tag}.log"
  local ok_file="${STATUS_DIR}/${tag}.ok"
  local fail_file="${STATUS_DIR}/${tag}.fail"
  rm -f "${ok_file}" "${fail_file}"

  echo "[RUN] ${tag} lr=${lr} tau=${mask_tau} alpha=${mask_alpha} ple=${plocal_epochs} adv=${adv_eps}/${adv_num_iter} rounds=${ROUNDS} purpose=${purpose}"
  set +e
  (
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    export PYTHONUNBUFFERED=1
    cd "${PFEDBA_ROOT}"
    "${PYTHON_BIN}" -u "${PFEDBA_MAIN}" \
      --dataset Cifar10 \
      --model resnet \
      --resnet_pretrained 0 \
      --algorithm FedCLCM \
      --batch_size 64 \
      --learning_rate "${lr}" \
      --lr_head "${lr}" \
      --num_global_iters "${ROUNDS}" \
      --local_epochs 1 \
      --plocal_epochs "${plocal_epochs}" \
      --numusers 10 \
      --times 1 \
      --seed 1 \
      --malclient 10 \
      --attack_start 30 \
      --poisoning_per_batch 1 \
      --attack_method attackall \
      --per_epoch 1 \
      --defense none \
      --eval_gap "${EVAL_GAP}" \
      --personalized_eval_gap 0 \
      --rt_beta 0.20 \
      --lambda_cl 0.05 \
      --aug_strength 0.10 \
      --mask_tau "${mask_tau}" \
      --mask_alpha "${mask_alpha}" \
      --enable_channel_mask 1 \
      --adv_eps "${adv_eps}" \
      --adv_num_iter "${adv_num_iter}" \
      --cosine_gate 0 \
      > "${log_file}" 2>&1
  )
  local rc=$?
  set -e

  if [[ "${rc}" -eq 0 ]]; then
    echo "ok $(date '+%F %T')" > "${ok_file}"
    echo "[OK] ${tag}"
  else
    echo "fail rc=${rc} $(date '+%F %T')" > "${fail_file}"
    echo "[FAIL] ${tag} rc=${rc}" >&2
    return "${rc}"
  fi
}

for cfg in "${CONFIGS[@]}"; do
  run_one "${cfg}"
done

echo "[READY] run_root=${RUN_ROOT}"
echo "[READY] manifest=${MANIFEST}"
