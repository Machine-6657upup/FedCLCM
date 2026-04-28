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
ROUNDS="${ROUNDS:-800}"
RUN_ROOT="${PROJECT_DIR}/runs/${RUN_TS}_pfedba_quick_story"
LOG_DIR="${RUN_ROOT}/train_logs"
STATUS_DIR="${RUN_ROOT}/status"
MANIFEST="${RUN_ROOT}/manifest.tsv"

mkdir -p "${LOG_DIR}" "${STATUS_DIR}"

cat > "${MANIFEST}" <<EOF
tag	gpu	algorithm	rounds	lr	lr_head	local_epochs	plocal_epochs	numusers	malclient	attack_start	poisoning_per_batch	mask_tau	mask_alpha	lambda_cl	purpose
PQ01_FEDAVG_PFEDBA	${GPU_ID}	FedAvg	${ROUNDS}	0.1	0.1	1	1	10	10	30	1	-	-	-	PFedBA quick sanity: no-defense FedAvg
PQ02_FEDREP_PFEDBA	${GPU_ID}	FedRep	${ROUNDS}	0.1	0.1	1	1	10	10	30	1	-	-	-	PFedBA quick sanity: FedRep vulnerability
PQ03_CLCM_T6A03_PFEDBA	${GPU_ID}	FedCLCM	${ROUNDS}	0.1	0.1	1	1	10	10	30	1	6.0	0.3	0.2	PFedBA quick defense: current static setting
PQ04_CLCM_T12A07_PFEDBA	${GPU_ID}	FedCLCM	${ROUNDS}	0.1	0.1	1	1	10	10	30	1	12.0	0.7	0.2	PFedBA quick defense: old low-ASR setting
EOF

if [[ ! -f "${PFEDBA_MAIN}" ]]; then
  echo "[FATAL] missing ${PFEDBA_MAIN}" >&2
  exit 1
fi

run_one() {
  local tag="$1"
  local algorithm="$2"
  local mask_tau="$3"
  local mask_alpha="$4"
  local lambda_cl="$5"
  local log_file="${LOG_DIR}/${tag}.log"
  local ok_file="${STATUS_DIR}/${tag}.ok"
  local fail_file="${STATUS_DIR}/${tag}.fail"

  echo "[RUN] ${tag} algo=${algorithm} rounds=${ROUNDS} gpu=${GPU_ID}"
  rm -f "${ok_file}" "${fail_file}"

  local -a cmd=(
    "${PYTHON_BIN}" -u "${PFEDBA_MAIN}"
    --dataset Cifar10
    --model resnet
    --resnet_pretrained 0
    --algorithm "${algorithm}"
    --batch_size 64
    --learning_rate 0.1
    --lr_head 0.1
    --num_global_iters "${ROUNDS}"
    --local_epochs 1
    --plocal_epochs 1
    --numusers 10
    --times 1
    --seed 1
    --malclient 10
    --attack_start 30
    --poisoning_per_batch 1
    --attack_method attackall
    --per_epoch 1
    --defense none
    --eval_gap 10
    --personalized_eval_gap 0
  )

  if [[ "${algorithm}" == "FedCLCM" ]]; then
    cmd+=(
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
  fi

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
    echo "[FAIL] ${tag} rc=${rc}"
    return "${rc}"
  fi
}

run_one PQ01_FEDAVG_PFEDBA FedAvg 0 0 0
run_one PQ02_FEDREP_PFEDBA FedRep 0 0 0
run_one PQ03_CLCM_T6A03_PFEDBA FedCLCM 6.0 0.3 0.2
run_one PQ04_CLCM_T12A07_PFEDBA FedCLCM 12.0 0.7 0.2

echo "[READY] ${RUN_ROOT}"
