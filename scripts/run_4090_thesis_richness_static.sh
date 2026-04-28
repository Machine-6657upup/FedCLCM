#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"

SRC_ROOT="${PROJECT_DIR}/src"
DATASET_ROOT="${SRC_ROOT}/dataset"
GEN_DIR="${DATASET_ROOT}/utils"
RAWDATA_ROOT="${DATASET_ROOT}/rawdata"
WRAPPER="${PROJECT_DIR}/scripts/generate_dataset_via_existing.py"
PYTHON_BIN="${PYTHON_BIN:-/home/fch/miniconda3/envs/fedclcm-static/bin/python}"
if [[ "${PYTHON_BIN}" == "python" ]]; then
  PYTHON_BIN="/home/fch/miniconda3/envs/fedclcm-static/bin/python"
fi
GPU_ID="${GPU_ID:-0}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-${RUN_TS}_thesis_richness_static_4090}"
RUN_ROOT="${PROJECT_DIR}/runs/${RUN_NAME}"
LOG_DIR="${RUN_ROOT}/train_logs"
STATUS_DIR="${RUN_ROOT}/status"
DATASET_LOG_DIR="${RUN_ROOT}/dataset_logs"
MANIFEST="${RUN_ROOT}/manifest.csv"
DATASET_MANIFEST="${RUN_ROOT}/dataset_manifest.csv"
ROUNDS="${ROUNDS:-800}"
EVAL_GAP="${EVAL_GAP:-10}"

mkdir -p "${LOG_DIR}" "${STATUS_DIR}" "${DATASET_LOG_DIR}"

ensure_dataset() {
  local dataset="$1"
  local generator="$2"
  local alpha="$3"
  local bdoor="$4"
  local adv="$5"
  local seed="$6"
  shift 6
  local log_file="${DATASET_LOG_DIR}/${dataset}.log"
  echo "[DATASET] ${dataset}"
  "${PYTHON_BIN}" "${WRAPPER}" \
    --generator "${GEN_DIR}/${generator}" \
    --dir-path "${DATASET_ROOT}/${dataset}" \
    --rawdata-path "${RAWDATA_ROOT}" \
    --num-clients 100 \
    --backdoor-rate "${bdoor}" \
    --adversary-num "${adv}" \
    --target-y 0 \
    --alpha "${alpha}" \
    --train-ratio 0.8 \
    --batch-size 10 \
    --partition dir \
    --niid \
    --balance \
    --generator-seed "${seed}" \
    "$@" > "${log_file}" 2>&1
  echo "${dataset},${generator},${alpha},${bdoor},${adv},${seed},${log_file}" >> "${DATASET_MANIFEST}"
}

echo "dataset,generator,alpha,bdoor,adv,seed,log_file" > "${DATASET_MANIFEST}"
ensure_dataset Cifar10_dir1.0_bdoor0.2_nclient_100_sig_adv10 generate_Cifar10_sig.py 1.0 0.2 10 42 --sig-delta 0.11764705882352941 --sig-f 6 --sig-label-mode dirty
ensure_dataset Cifar10_dir0.5_bdoor0.1_nclient_100_blend_adv10 generate_Cifar10_blend.py 0.5 0.1 10 42 --blend-alpha 0.2
ensure_dataset Cifar10_dir0.5_bdoor0.1_nclient_100_sig_adv10 generate_Cifar10_sig.py 0.5 0.1 10 42 --sig-delta 0.11764705882352941 --sig-f 6 --sig-label-mode dirty
ensure_dataset Cifar10_dir0.5_bdoor0.0_nclient_100_badnet_adv0 generate_Cifar10_badnet.py 0.5 0.0 0 42

cat > "${MANIFEST}" <<CSV
tag,gpu,algorithm,dataset,rounds,jr,lr,lr_head,local_epochs,adv_clients,purpose
R01_SIG_DIR10,${GPU_ID},FedCLCM,Cifar10_dir1.0_bdoor0.2_nclient_100_sig_adv10,${ROUNDS},0.1,0.1,0.1,1,10,heterogeneity alpha=1.0 for SIG
R02_BLEND_BDOOR01,${GPU_ID},FedCLCM,Cifar10_dir0.5_bdoor0.1_nclient_100_blend_adv10,${ROUNDS},0.1,0.1,0.1,1,10,attack strength bdoor=0.1 for Blend
R03_SIG_BDOOR01,${GPU_ID},FedCLCM,Cifar10_dir0.5_bdoor0.1_nclient_100_sig_adv10,${ROUNDS},0.1,0.1,0.1,1,10,attack strength bdoor=0.1 for SIG
R04_CLEAN_FEDAVG,${GPU_ID},FedAvg,Cifar10_dir0.5_bdoor0.0_nclient_100_badnet_adv0,${ROUNDS},0.1,0.1,0.1,1,0,clean utility upper bound FedAvg
R05_CLEAN_FEDREP,${GPU_ID},FedRep,Cifar10_dir0.5_bdoor0.0_nclient_100_badnet_adv0,${ROUNDS},0.1,0.1,0.1,1,0,clean utility upper bound FedRep
R06_CLEAN_CLCM,${GPU_ID},FedCLCM,Cifar10_dir0.5_bdoor0.0_nclient_100_badnet_adv0,${ROUNDS},0.1,0.1,0.1,1,0,clean utility upper bound FedCLCM
CSV

run_one() {
  local tag="$1"
  local algorithm="$2"
  local dataset="$3"
  local adv_clients="$4"
  local log_file="${LOG_DIR}/${tag}.log"
  local ok_file="${STATUS_DIR}/${tag}.ok"
  local fail_file="${STATUS_DIR}/${tag}.fail"
  echo "[RUN] ${tag} algo=${algorithm} dataset=${dataset}"
  rm -f "${ok_file}" "${fail_file}"

  local -a cmd=(
    "${PYTHON_BIN}" -u main.py
    -dev cuda -did "${GPU_ID}"
    -data "${dataset}"
    -m ResNet18
    -algo "${algorithm}"
    -ncl 10 -nc 100 -jr 0.1 -lbs 64
    -lr 0.1 -lr_head 0.1 -ls 1 -pls 1
    -gr "${ROUNDS}" -eg "${EVAL_GAP}"
    -go "${tag}"
    --num_adv_clients "${adv_clients}"
  )
  if [[ "${algorithm}" == "FedCLCM" ]]; then
    cmd+=(--rt_beta 0.2 --lambda_cl 0.2 --aug_strength 0.1 --mask_tau 6.0 --mask_alpha 0.3 --enable_channel_mask true --adv_eps 0.0 --adv_num_iter 0)
  fi

  set +e
  (cd "${SRC_ROOT}" && "${cmd[@]}" > "${log_file}" 2>&1)
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

run_one R01_SIG_DIR10 FedCLCM Cifar10_dir1.0_bdoor0.2_nclient_100_sig_adv10 10
run_one R02_BLEND_BDOOR01 FedCLCM Cifar10_dir0.5_bdoor0.1_nclient_100_blend_adv10 10
run_one R03_SIG_BDOOR01 FedCLCM Cifar10_dir0.5_bdoor0.1_nclient_100_sig_adv10 10
run_one R04_CLEAN_FEDAVG FedAvg Cifar10_dir0.5_bdoor0.0_nclient_100_badnet_adv0 0
run_one R05_CLEAN_FEDREP FedRep Cifar10_dir0.5_bdoor0.0_nclient_100_badnet_adv0 0
run_one R06_CLEAN_CLCM FedCLCM Cifar10_dir0.5_bdoor0.0_nclient_100_badnet_adv0 0

echo "[READY] ${RUN_ROOT}"
