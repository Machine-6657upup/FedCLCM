#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"

DATASET_ROOT="${PROJECT_DIR}/dataset"
RAWDATA_ROOT="${DATASET_ROOT}/rawdata"
LOG_DIR="${PROJECT_DIR}/thesis_log/dataset_prep"
RUN_TS="$(timestamp)"
MANIFEST="${LOG_DIR}/dataset_manifest_${RUN_TS}.csv"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

ensure_dir "${DATASET_ROOT}"
ensure_dir "${RAWDATA_ROOT}"
ensure_dir "${LOG_DIR}"

echo "dataset_name,generator,num_clients,backdoor_rate,adversary_num,target_y,alpha,niid,balance,partition,force_rebuild" > "${MANIFEST}"

echo "[1/3] Create trigger assets if missing..."
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/create_trigger_assets.py"

run_dataset() {
  local dataset_name="$1"
  local generator_name="$2"
  local num_clients="$3"
  local backdoor_rate="$4"
  local adversary_num="$5"
  local target_y="$6"
  local alpha="$7"
  local niid="$8"
  local balance="$9"
  local partition="${10}"

  local -a cmd=(
    "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/generate_dataset_via_existing.py"
    --generator "${PROJECT_DIR}/dataset/utils/${generator_name}"
    --dir-path "${DATASET_ROOT}/${dataset_name}"
    --rawdata-path "${RAWDATA_ROOT}"
    --num-clients "${num_clients}"
    --backdoor-rate "${backdoor_rate}"
    --adversary-num "${adversary_num}"
    --target-y "${target_y}"
    --alpha "${alpha}"
    --train-ratio 0.8
    --batch-size 10
    --partition "${partition}"
  )

  if [[ "${niid}" == "1" ]]; then
    cmd+=(--niid)
  fi
  if [[ "${balance}" == "1" ]]; then
    cmd+=(--balance)
  fi
  if [[ "${FORCE_REBUILD}" == "1" ]]; then
    cmd+=(--force-rebuild)
  fi

  echo ">>> Generating ${dataset_name}"
  "${cmd[@]}"
  echo "${dataset_name},${generator_name},${num_clients},${backdoor_rate},${adversary_num},${target_y},${alpha},${niid},${balance},${partition},${FORCE_REBUILD}" >> "${MANIFEST}"
}

echo "[2/3] Generate chapter-4 core datasets..."
run_dataset "Cifar10_dir0.5_nclient_40_badpfl_adv5" "generate_Cifar10_badpfl.py" 40 0.0 5 0 0.5 1 1 dir
run_dataset "FashionMNIST_dir0.5_bdoor0.0_nclient_100_badnet_adv0" "generate_FashionMNIST_badnet.py" 100 0.0 0 1 0.5 1 1 dir
run_dataset "Cifar10_dir0.5_bdoor0.2_nclient_40_badnet_adv5" "generate_Cifar10_badnet.py" 40 0.2 5 0 0.5 1 1 dir
run_dataset "Cifar10_dir0.5_bdoor0.2_nclient_40_blend_adv5" "generate_Cifar10_blend.py" 40 0.2 5 0 0.5 1 1 dir
run_dataset "Cifar10_dir0.5_bdoor0.2_nclient_40_sig_adv5" "generate_Cifar10_sig.py" 40 0.2 5 0 0.5 1 1 dir
run_dataset "FashionMNIST_dir0.5_bdoor0.2_nclient_40_badnet_adv5" "generate_FashionMNIST_badnet.py" 40 0.2 5 0 0.5 1 1 dir
run_dataset "FashionMNIST_dir0.5_bdoor0.2_nclient_40_blend_adv5" "generate_FashionMNIST_blend.py" 40 0.2 5 0 0.5 1 1 dir
run_dataset "FashionMNIST_dir0.5_bdoor0.2_nclient_40_sig_adv5" "generate_FashionMNIST_sig.py" 40 0.2 5 0 0.5 1 1 dir
run_dataset "MNIST_dir0.5_bdoor0.5_nclient_40_badnet_adv5" "generate_MNIST_badnet.py" 40 0.5 5 0 0.5 1 1 dir
run_dataset "MNIST_dir0.5_bdoor0.5_nclient_40_blend_adv5" "generate_MNIST_blend.py" 40 0.5 5 0 0.5 1 1 dir
run_dataset "MNIST_dir0.5_bdoor0.5_nclient_40_sig_adv5" "generate_MNIST_sig.py" 40 0.5 5 0 0.5 1 1 dir

echo "[3/3] Generate heterogeneity variants used by chapter-4 robustness scripts..."
run_dataset "Cifar10_dir0.2_bdoor0.2_nclient_40_badnet_adv5" "generate_Cifar10_badnet.py" 40 0.2 5 0 0.2 1 1 dir
run_dataset "Cifar10_dir0.8_bdoor0.2_nclient_40_badnet_adv5" "generate_Cifar10_badnet.py" 40 0.2 5 0 0.8 1 1 dir
run_dataset "Cifar10_iid_bdoor0.2_nclient_40_badnet_adv5" "generate_Cifar10_badnet.py" 40 0.2 5 0 0.5 0 1 -

echo "Dataset manifest written to: ${MANIFEST}"
