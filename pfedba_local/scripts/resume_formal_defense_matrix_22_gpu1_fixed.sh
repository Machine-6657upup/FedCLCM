#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/huangtu/PFL_clean_workspace/root_static/pfedba_local"
PYTHON_BIN="${PYTHON_BIN:-/home/huangtu/miniconda3/envs/torch/bin/python}"
MAIN_PY="${ROOT_DIR}/main.py"
LOGDIR="${LOGDIR:-${ROOT_DIR}/log/formal_defense_matrix_22_20260419_213225}"
STATUS_DIR="${STATUS_DIR:-${LOGDIR}/status}"
GPU="${GPU:-1}"

mkdir -p "${LOGDIR}" "${STATUS_DIR}"

COMMON_ARGS=(
  --dataset Cifar10
  --model resnet
  --resnet_pretrained 0
  --learning_rate 0.1
  --lr_head 0.1
  --plocal_epochs 1
  --numusers 10
  --batch_size 64
  --attack_start 30
  --attack_method attackall
  --poisoning_per_batch 1
  --defense none
  --per_epoch 1
  --malclient 10
  --times 1
)

require_worker_gpu() {
  local check_out
  check_out="$(CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" - <<'PY'
import sys
import torch
ok = torch.cuda.is_available()
count = torch.cuda.device_count() if ok else 0
name = torch.cuda.get_device_name(0) if ok and count >= 1 else "N/A"
print(f"cuda_available={ok}")
print(f"device_count={count}")
print(f"device0={name}")
sys.exit(0 if ok and count == 1 else 1)
PY
)"
  local rc=$?
  printf '%s\n' "${check_out}"
  if [[ "${rc}" -ne 0 ]]; then
    echo "[FATAL] gpu${GPU} resume queue does not have exactly one visible CUDA device." >&2
    return 1
  fi
}

run_one() {
  local tag="$1"
  local algorithm="$2"
  local local_epochs="$3"
  local num_global_iters="$4"
  shift 4

  local log_file="${LOGDIR}/${tag}.log"
  local meta_file="${LOGDIR}/${tag}.meta.log"
  local running_file="${STATUS_DIR}/${tag}.running"
  local ok_file="${STATUS_DIR}/${tag}.ok"
  local fail_file="${STATUS_DIR}/${tag}.fail"
  local mpl_dir="/tmp/mpl_resume_gpu1_${tag}"

  rm -f "${ok_file}" "${fail_file}"
  : > "${running_file}"
  mkdir -p "${mpl_dir}"

  {
    echo "=================================================="
    echo "[START] ${tag}"
    echo "TIME=$(date '+%F %T')"
    echo "GPU=${GPU}"
    echo "ALGORITHM=${algorithm}"
    echo "LOCAL_EPOCHS=${local_epochs}"
    echo "NUM_GLOBAL_ITERS=${num_global_iters}"
    echo "LOG=${log_file}"
    printf 'CMD=%q ' "${PYTHON_BIN}" -u "${MAIN_PY}" "${COMMON_ARGS[@]}" --algorithm "${algorithm}" --local_epochs "${local_epochs}" --num_global_iters "${num_global_iters}" "$@"
    echo
    echo "[GPU_CHECK]"
    require_worker_gpu
    echo "=================================================="
  } > "${meta_file}"

  set +e
  (
    export CUDA_VISIBLE_DEVICES="${GPU}"
    export PYTHONUNBUFFERED=1
    export MPLCONFIGDIR="${mpl_dir}"
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" -u "${MAIN_PY}" \
      "${COMMON_ARGS[@]}" \
      --algorithm "${algorithm}" \
      --local_epochs "${local_epochs}" \
      --num_global_iters "${num_global_iters}" \
      "$@" > "${log_file}" 2>&1
  )
  local rc=$?
  set -e

  rm -f "${running_file}"
  if [[ "${rc}" -eq 0 ]]; then
    : > "${ok_file}"
  else
    : > "${fail_file}"
  fi

  {
    echo "[END] ${tag} RC=${rc} TIME=$(date '+%F %T')"
    echo "=================================================="
  } >> "${meta_file}"

  return "${rc}"
}

run_one E02 FedRepPFLALP 10 400 --alp_use_cluster 1 --alp_use_purify 0 --cluster_max_k 4
run_one E10 PFLALP 10 400 --purify_beta 1500 --purify_rounds 2 --cluster_max_k 4
