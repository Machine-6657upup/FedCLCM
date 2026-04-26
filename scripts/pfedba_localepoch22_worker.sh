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
  local local_epochs="$4"
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
    echo "LOCAL_EPOCHS=${local_epochs}"
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
    --local_epochs "${local_epochs}" \
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
    run_one 0 E01 FedRep 1 baseline
    run_one 0 E05 FedRT 5 baseline
    run_one 0 E09 FedRPD 10 full
    run_one 0 E13 FedRPD 10 no_purify
    run_one 0 E17 FedRPD 10 no_distill
    run_one 0 E21 FedRPD 10 purify2
    ;;
  gpu1)
    run_one 1 E02 FedRep 5 baseline
    run_one 1 E06 FedRT 10 baseline
    run_one 1 E10 FedRPD 20 full
    run_one 1 E14 FedRPD 20 no_purify
    run_one 1 E18 FedRPD 20 no_distill
    run_one 1 E22 FedRPD 20 purify2
    ;;
  gpu2)
    run_one 2 E03 FedRep 10 baseline
    run_one 2 E07 FedRPD 1 full
    run_one 2 E11 FedRPD 1 no_purify
    run_one 2 E15 FedRPD 1 no_distill
    run_one 2 E19 FedRPD 1 purify2
    ;;
  gpu3)
    run_one 3 E04 FedRT 1 baseline
    run_one 3 E08 FedRPD 5 full
    run_one 3 E12 FedRPD 5 no_purify
    run_one 3 E16 FedRPD 5 no_distill
    run_one 3 E20 FedRPD 5 purify2
    ;;
  *)
    echo "unknown queue: ${QUEUE}"
    exit 2
    ;;
esac
