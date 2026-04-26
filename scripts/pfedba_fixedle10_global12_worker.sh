#!/usr/bin/env bash
set -u

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <gpu_queue> <logdir> <status_dir> <run_ts>"
  exit 2
fi

QUEUE="$1"
LOGDIR="$2"
STATUS_DIR="$3"
RUN_TS="$4"

PFEDBA_ROOT="/home/huangtu/PFL_Backdoor_Defense/PFedBA"
PYTHON_BIN="${PYTHON_BIN:-/home/huangtu/miniconda3/envs/torch/bin/python}"
MAIN_PY="${PFEDBA_ROOT}/main.py"

mkdir -p "${LOGDIR}" "${STATUS_DIR}"
cd "${PFEDBA_ROOT}" || exit 1

COMMON_ARGS=(
  --dataset Cifar10
  --model resnet
  --resnet_pretrained 0
  --learning_rate 0.1
  --lr_head 0.1
  --plocal_epochs 1
  --local_epochs 10
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

FEDRT_ARGS=(
  --rt_beta 0.10
  --adv_eps 0.10
  --adv_num_iter 5
  --aug_strength 0.10
)

FEDRPD_ARGS=(
  --rt_beta 0.10
  --adv_eps 0.10
  --adv_num_iter 5
  --aug_strength 0.10
  --purify_beta 1500
  --purify_rounds 1
  --distill_gamma 1.0
  --distill_weight 1.0
)

run_one() {
  local gpu="$1"
  local tag="$2"
  local algo="$3"
  local global_iters="$4"
  local variant="$5"

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
    echo "RUN_TS=${RUN_TS}"
    echo "GPU=${gpu}"
    echo "ALGORITHM=${algo}"
    echo "LOCAL_EPOCHS=10"
    echo "PLOCAL_EPOCHS=1"
    echo "GLOBAL_ITERS=${global_iters}"
    echo "VARIANT=${variant}"
    echo "TIME=$(date '+%F %T')"
    echo "LOG=${log_file}"
    echo "=================================================="
  } >> "${meta_file}"

  local extra_args=()
  if [[ "${algo}" == "FedRT" ]]; then
    extra_args=("${FEDRT_ARGS[@]}")
  elif [[ "${algo}" == "FedRPD" ]]; then
    extra_args=("${FEDRPD_ARGS[@]}")
    case "${variant}" in
      no_purify)
        extra_args+=(--purify_beta 0)
        ;;
      no_distill)
        extra_args+=(--distill_weight 0)
        ;;
      purify2)
        extra_args+=(--purify_rounds 2)
        ;;
    esac
  fi

  env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" -u "${MAIN_PY}" \
    "${COMMON_ARGS[@]}" \
    --algorithm "${algo}" \
    --num_global_iters "${global_iters}" \
    "${extra_args[@]}" \
    > "${log_file}" 2>&1
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

  return "${rc}"
}

case "${QUEUE}" in
  gpu0)
    run_one 0 G01 FedRep 250 baseline
    run_one 0 G05 FedRPD 250 no_purify
    run_one 0 G09 FedRT 300 baseline
    ;;
  gpu1)
    run_one 1 G02 FedRT 250 baseline
    run_one 1 G06 FedRPD 250 no_distill
    run_one 1 G10 FedRPD 300 full
    ;;
  gpu2)
    run_one 2 G03 FedRPD 250 full
    run_one 2 G07 FedRPD 300 purify2
    run_one 2 G11 FedRPD 300 no_purify
    ;;
  gpu3)
    run_one 3 G04 FedRPD 250 purify2
    run_one 3 G08 FedRep 300 baseline
    run_one 3 G12 FedRPD 300 no_distill
    ;;
  *)
    echo "unknown queue: ${QUEUE}"
    exit 2
    ;;
esac
