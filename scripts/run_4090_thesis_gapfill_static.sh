#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"

SRC_ROOT="${PROJECT_DIR}/src"
if [[ -z "${PYTHON_BIN:-}" || "${PYTHON_BIN}" == "python" ]]; then
  PYTHON_BIN="/home/fch/miniconda3/envs/fedclcm-static/bin/python"
fi
GPU_ID="${GPU_ID:-0}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-${RUN_TS}_thesis_gapfill_static_4090}"
RUN_ROOT="${PROJECT_DIR}/runs/${RUN_NAME}"
LOG_DIR="${RUN_ROOT}/train_logs"
STATUS_DIR="${RUN_ROOT}/status"
MANIFEST="${RUN_ROOT}/manifest.csv"
ROUNDS="${ROUNDS:-800}"
EVAL_GAP="${EVAL_GAP:-10}"

mkdir -p "${LOG_DIR}" "${STATUS_DIR}"

cat > "${MANIFEST}" <<CSV
tag,gpu,dataset,attack,algorithm,rounds,join_ratio,lr,lr_head,local_epochs,plocal_epochs,adv_clients,rt_beta,lambda_cl,mask_tau,mask_alpha,purpose
G01_BADNET_DIR10,${GPU_ID},Cifar10_dir1.0_bdoor0.2_nclient_100_badnet_adv10,badnet,FedCLCM,${ROUNDS},0.1,0.1,0.1,1,1,10,0.2,0.2,6.0,0.3,rerun failed H02 for heterogeneity alpha=1.0
G02_SIG_BDOOR03,${GPU_ID},Cifar10_dir0.5_bdoor0.3_nclient_100_sig_adv10,sig,FedCLCM,${ROUNDS},0.1,0.1,0.1,1,1,10,0.2,0.2,6.0,0.3,rerun failed P04 for attack strength bdoor=0.3
G03_BADNET_LAMBDA05,${GPU_ID},Cifar10_dir0.5_bdoor0.2_nclient_100_badnet_adv10,badnet,FedCLCM,${ROUNDS},0.1,0.1,0.1,1,1,10,0.2,0.5,6.0,0.3,contrastive sensitivity lambda_cl=0.5
G04_BLEND_DIR10,${GPU_ID},Cifar10_dir1.0_bdoor0.2_nclient_100_blend_adv10,blend,FedCLCM,${ROUNDS},0.1,0.1,0.1,1,1,10,0.2,0.2,6.0,0.3,heterogeneity alpha=1.0 for Blend
CSV

require_dataset() {
  local dataset="$1"
  if [[ ! -f "${SRC_ROOT}/dataset/${dataset}/config.json" ]]; then
    echo "[FATAL] missing dataset: ${SRC_ROOT}/dataset/${dataset}/config.json" >&2
    exit 1
  fi
}

run_one() {
  local tag="$1"
  local dataset="$2"
  local adv_clients="$3"
  local lambda_cl="$4"
  local log_file="${LOG_DIR}/${tag}.log"
  local ok_file="${STATUS_DIR}/${tag}.ok"
  local fail_file="${STATUS_DIR}/${tag}.fail"

  require_dataset "${dataset}"
  rm -f "${ok_file}" "${fail_file}"
  echo "[RUN] ${tag} dataset=${dataset} rounds=${ROUNDS} gpu=${GPU_ID} lambda_cl=${lambda_cl}"

  local -a cmd=(
    "${PYTHON_BIN}" -u main.py
    -dev cuda
    -did "${GPU_ID}"
    -data "${dataset}"
    -m ResNet18
    -algo FedCLCM
    -ncl 10
    -nc 100
    -jr 0.1
    -lbs 64
    -lr 0.1
    -lr_head 0.1
    -ls 1
    -pls 1
    -gr "${ROUNDS}"
    -eg "${EVAL_GAP}"
    -go "${tag}"
    --num_adv_clients "${adv_clients}"
    --rt_beta 0.2
    --lambda_cl "${lambda_cl}"
    --aug_strength 0.1
    --mask_tau 6.0
    --mask_alpha 0.3
    --enable_channel_mask true
    --adv_eps 0.0
    --adv_num_iter 0
  )

  set +e
  (
    cd "${SRC_ROOT}"
    export PYTHONUNBUFFERED=1
    "${cmd[@]}" > "${log_file}" 2>&1
  )
  local rc=$?
  set -e

  if [[ "${rc}" -eq 0 ]]; then
    echo "ok rc=0 $(date '+%F %T')" > "${ok_file}"
    echo "[OK] ${tag}"
  else
    echo "fail rc=${rc} $(date '+%F %T')" > "${fail_file}"
    echo "[FAIL] ${tag} rc=${rc}" >&2
    return "${rc}"
  fi
}

run_one G01_BADNET_DIR10 Cifar10_dir1.0_bdoor0.2_nclient_100_badnet_adv10 10 0.2
run_one G02_SIG_BDOOR03 Cifar10_dir0.5_bdoor0.3_nclient_100_sig_adv10 10 0.2
run_one G03_BADNET_LAMBDA05 Cifar10_dir0.5_bdoor0.2_nclient_100_badnet_adv10 10 0.5
run_one G04_BLEND_DIR10 Cifar10_dir1.0_bdoor0.2_nclient_100_blend_adv10 10 0.2

echo "[READY] ${RUN_ROOT}"
