#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"

PFEDBA_ROOT="${PROJECT_DIR}/pfedba_local"
PFEDBA_MAIN="${PFEDBA_ROOT}/main.py"
PYTHON_BIN="${PYTHON_BIN:-/home/fch/miniconda3/envs/fedclcm-static/bin/python}"
if [[ "${PYTHON_BIN}" == "python" ]]; then
  PYTHON_BIN="/home/fch/miniconda3/envs/fedclcm-static/bin/python"
fi
GPU_ID="${GPU_ID:-0}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-${RUN_TS}_pfedba_clcm_sweep_4090}"
RUN_ROOT="${PROJECT_DIR}/runs/${RUN_NAME}"
LOG_DIR="${RUN_ROOT}/train_logs"
STATUS_DIR="${RUN_ROOT}/status"
MANIFEST="${RUN_ROOT}/manifest.csv"
ROUNDS="${ROUNDS:-800}"
EVAL_GAP="${EVAL_GAP:-10}"

mkdir -p "${LOG_DIR}" "${STATUS_DIR}"

cat > "${MANIFEST}" <<CSV
tag,gpu,algorithm,rounds,lr,lr_head,local_epochs,plocal_epochs,mask_tau,mask_alpha,lambda_cl,purpose
PB01_T8A05,${GPU_ID},FedCLCM,${ROUNDS},0.1,0.1,1,1,8.0,0.5,0.2,PFedBA CLCM middle mask strength
PB02_T10A07,${GPU_ID},FedCLCM,${ROUNDS},0.1,0.1,1,1,10.0,0.7,0.2,PFedBA CLCM stronger mask
PB03_T12A09,${GPU_ID},FedCLCM,${ROUNDS},0.1,0.1,1,1,12.0,0.9,0.2,PFedBA CLCM very strong mask
PB04_T12A07_LAM05,${GPU_ID},FedCLCM,${ROUNDS},0.1,0.1,1,1,12.0,0.7,0.5,PFedBA stronger contrastive loss
PB05_T12A07_LR005,${GPU_ID},FedCLCM,${ROUNDS},0.05,0.05,1,1,12.0,0.7,0.2,PFedBA lower learning rate
PB06_T12A07_LR02,${GPU_ID},FedCLCM,${ROUNDS},0.2,0.2,1,1,12.0,0.7,0.2,PFedBA higher learning rate
PB07_T12A07_LE5,${GPU_ID},FedCLCM,${ROUNDS},0.1,0.1,5,1,12.0,0.7,0.2,PFedBA more local base epochs
PB08_T12A07_PLE5,${GPU_ID},FedCLCM,${ROUNDS},0.1,0.1,1,5,12.0,0.7,0.2,PFedBA more personalized head epochs
PB09_T6A03_LE5,${GPU_ID},FedCLCM,${ROUNDS},0.1,0.1,5,1,6.0,0.3,0.2,PFedBA current static mask with more local epochs
PB10_T8A05_LE5,${GPU_ID},FedCLCM,${ROUNDS},0.1,0.1,5,1,8.0,0.5,0.2,PFedBA middle mask with more local epochs
PB11_T12A07_LE10,${GPU_ID},FedCLCM,${ROUNDS},0.1,0.1,10,1,12.0,0.7,0.2,PFedBA long local epoch stress test
PB12_T12A07_PLE10,${GPU_ID},FedCLCM,${ROUNDS},0.1,0.1,1,10,12.0,0.7,0.2,PFedBA long personalized epoch stress test
CSV

run_one() {
  local tag="$1"
  local lr="$2"
  local lr_head="$3"
  local local_epochs="$4"
  local plocal_epochs="$5"
  local mask_tau="$6"
  local mask_alpha="$7"
  local lambda_cl="$8"
  local log_file="${LOG_DIR}/${tag}.log"
  local ok_file="${STATUS_DIR}/${tag}.ok"
  local fail_file="${STATUS_DIR}/${tag}.fail"

  echo "[RUN] ${tag} lr=${lr} le=${local_epochs} ple=${plocal_epochs} tau=${mask_tau} alpha=${mask_alpha} lambda=${lambda_cl}"
  rm -f "${ok_file}" "${fail_file}"

  local -a cmd=(
    "${PYTHON_BIN}" -u "${PFEDBA_MAIN}"
    --dataset Cifar10
    --model resnet
    --resnet_pretrained 0
    --algorithm FedCLCM
    --batch_size 64
    --learning_rate "${lr}"
    --lr_head "${lr_head}"
    --num_global_iters "${ROUNDS}"
    --local_epochs "${local_epochs}"
    --plocal_epochs "${plocal_epochs}"
    --numusers 10
    --times 1
    --seed 1
    --malclient 10
    --attack_start 30
    --poisoning_per_batch 1
    --attack_method attackall
    --per_epoch 1
    --defense none
    --eval_gap "${EVAL_GAP}"
    --personalized_eval_gap 0
    --rt_beta 0.20
    --lambda_cl "${lambda_cl}"
    --aug_strength 0.10
    --mask_tau "${mask_tau}"
    --mask_alpha "${mask_alpha}"
    --enable_channel_mask 1
    --adv_eps 0
    --adv_num_iter 0
    --cosine_gate 0
  )

  set +e
  (
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    export PYTHONUNBUFFERED=1
    cd "${PFEDBA_ROOT}"
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
  fi
}

run_one PB01_T8A05 0.1 0.1 1 1 8.0 0.5 0.2
run_one PB02_T10A07 0.1 0.1 1 1 10.0 0.7 0.2
run_one PB03_T12A09 0.1 0.1 1 1 12.0 0.9 0.2
run_one PB04_T12A07_LAM05 0.1 0.1 1 1 12.0 0.7 0.5
run_one PB05_T12A07_LR005 0.05 0.05 1 1 12.0 0.7 0.2
run_one PB06_T12A07_LR02 0.2 0.2 1 1 12.0 0.7 0.2
run_one PB07_T12A07_LE5 0.1 0.1 5 1 12.0 0.7 0.2
run_one PB08_T12A07_PLE5 0.1 0.1 1 5 12.0 0.7 0.2
run_one PB09_T6A03_LE5 0.1 0.1 5 1 6.0 0.3 0.2
run_one PB10_T8A05_LE5 0.1 0.1 5 1 8.0 0.5 0.2
run_one PB11_T12A07_LE10 0.1 0.1 10 1 12.0 0.7 0.2
run_one PB12_T12A07_PLE10 0.1 0.1 1 10 12.0 0.7 0.2

echo "[READY] ${RUN_ROOT}"
