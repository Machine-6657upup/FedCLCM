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
RUN_NAME="${RUN_NAME:-${RUN_TS}_pfedba_lowcl_existing_entry_4090}"
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
  local purpose="$5"
  CONFIGS+=("${tag}|${lr}|${mask_tau}|${mask_alpha}|${purpose}")
}

for lr in 0.03 0.05 0.08; do
  lr_tag="${lr/./}"
  add_config "PF_LR${lr_tag}_T50A03_LC005" "${lr}" 5.0 0.3 "pfedba_lowcl_lr${lr}_tau5_alpha03"
  add_config "PF_LR${lr_tag}_T80A05_LC005" "${lr}" 8.0 0.5 "pfedba_lowcl_lr${lr}_tau8_alpha05"
  add_config "PF_LR${lr_tag}_T100A07_LC005" "${lr}" 10.0 0.7 "pfedba_lowcl_lr${lr}_tau10_alpha07"
  add_config "PF_LR${lr_tag}_T120A07_LC005" "${lr}" 12.0 0.7 "pfedba_lowcl_lr${lr}_tau12_alpha07"
done

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

echo -e "tag\tlr\tlr_head\tlambda_cl\tmask_tau\tmask_alpha\trounds\tentry\tpurpose" > "${MANIFEST}"
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r tag lr tau alpha purpose <<< "${cfg}"
  echo -e "${tag}\t${lr}\t${lr}\t0.05\t${tau}\t${alpha}\t${ROUNDS}\tpfedba_local/main.py\t${purpose}" >> "${MANIFEST}"
done

run_one() {
  local cfg="$1"
  IFS='|' read -r tag lr mask_tau mask_alpha purpose <<< "${cfg}"
  local log_file="${LOG_DIR}/${tag}.log"
  local ok_file="${STATUS_DIR}/${tag}.ok"
  local fail_file="${STATUS_DIR}/${tag}.fail"
  rm -f "${ok_file}" "${fail_file}"

  echo "[RUN] ${tag} lr=${lr} tau=${mask_tau} alpha=${mask_alpha} rounds=${ROUNDS} purpose=${purpose}"
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
      --plocal_epochs 1 \
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
      --adv_eps 0 \
      --adv_num_iter 0 \
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
