#!/usr/bin/env bash
set -u

PFEDBA_ROOT="/home/huangtu/PFL_Backdoor_Defense/PFedBA"
PYTHON_BIN="${PYTHON_BIN:-/home/huangtu/miniconda3/envs/torch/bin/python}"
MAIN_PY="${PFEDBA_ROOT}/main.py"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-${PFEDBA_ROOT}/log/fedrt_nohup12_${RUN_TS}}"
STATUS_DIR="${LOGDIR}/status"

mkdir -p "${LOGDIR}" "${STATUS_DIR}"
cd "${PFEDBA_ROOT}" || exit 1

common_args=(
  --dataset Cifar10
  --model resnet
  --resnet_pretrained 0
  --algorithm FedRT
  --learning_rate 0.1
  --lr_head 0.1
  --plocal_epochs 1
  --local_epochs 20
  --num_global_iters 150
  --numusers 10
  --batch_size 64
  --attack_start 30
  --attack_method attackall
  --poisoning_per_batch 5
  --defense none
  --per_epoch 1
  --malclient 10
  --times 1
)

run_one() {
  local gpu="$1"
  local tag="$2"
  shift 2

  local log_file="${LOGDIR}/${tag}.log"
  local meta_file="${LOGDIR}/${tag}.meta.log"
  local running_file="${STATUS_DIR}/${tag}.running"
  local ok_file="${STATUS_DIR}/${tag}.ok"
  local fail_file="${STATUS_DIR}/${tag}.fail"

  rm -f "${ok_file}" "${fail_file}"
  touch "${running_file}"

  {
    echo "=================================================="
    echo "[START] ${tag}"
    echo "GPU=${gpu}"
    echo "TIME=$(date '+%F %T')"
    echo "LOG=${log_file}"
    echo "=================================================="
  } >> "${meta_file}"

  env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" -u "${MAIN_PY}" \
    "${common_args[@]}" \
    "$@" > "${log_file}" 2>&1
  local rc=$?

  rm -f "${running_file}"
  if [[ "${rc}" -eq 0 ]]; then
    touch "${ok_file}"
  else
    touch "${fail_file}"
  fi

  {
    echo "[END] ${tag} RC=${rc} TIME=$(date '+%F %T')"
    echo "=================================================="
  } >> "${meta_file}"
}

queue_gpu0() {
  run_one 0 F01 --rt_beta 0.10 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10
  run_one 0 F05 --rt_beta 0.10 --adv_eps 0.05 --adv_num_iter 3 --aug_strength 0.10
  run_one 0 F09 --rt_beta 0.05 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10
}

queue_gpu1() {
  run_one 1 F02 --rt_beta 0.10 --adv_eps 0.00 --adv_num_iter 0 --aug_strength 0.10
  run_one 1 F06 --rt_beta 0.10 --adv_eps 0.08 --adv_num_iter 5 --aug_strength 0.10
  run_one 1 F10 --rt_beta 0.08 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10
}

queue_gpu2() {
  run_one 2 F03 --rt_beta 0.00 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10
  run_one 2 F07 --rt_beta 0.10 --adv_eps 0.12 --adv_num_iter 5 --aug_strength 0.10
  run_one 2 F11 --rt_beta 0.12 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10
}

queue_gpu3() {
  run_one 3 F04 --rt_beta 0.10 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.00
  run_one 3 F08 --rt_beta 0.10 --adv_eps 0.15 --adv_num_iter 7 --aug_strength 0.10
  run_one 3 F12 --rt_beta 0.15 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10
}

echo "PFEDBA_ROOT=${PFEDBA_ROOT}"
echo "RUN_TS=${RUN_TS}"
echo "LOGDIR=${LOGDIR}"
echo "STATUS_DIR=${STATUS_DIR}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "MAIN_PY=${MAIN_PY}"

queue_gpu0 &
PID0=$!
queue_gpu1 &
PID1=$!
queue_gpu2 &
PID2=$!
queue_gpu3 &
PID3=$!

wait "${PID0}" "${PID1}" "${PID2}" "${PID3}"
