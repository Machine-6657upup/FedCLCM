#!/usr/bin/env bash
set -euo pipefail

PFEDBA_ROOT="${PFEDBA_ROOT:-/home/huangtu/PFL_Backdoor_Defense/PFedBA}"
PYTHON_BIN="${PYTHON_BIN:-/home/huangtu/miniconda3/envs/torch/bin/python}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-${PFEDBA_ROOT}/log/fedrpd_localepoch24_${RUN_TS}}"

mkdir -p "${LOGDIR}"

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

MANIFEST=$'L01|0|FedRep|1|baseline\nL02|1|FedRep|5|baseline\nL03|2|FedRep|10|baseline\nL04|3|FedRep|20|baseline\nL05|0|FedRT|1|baseline\nL06|1|FedRT|5|baseline\nL07|2|FedRT|10|baseline\nL08|3|FedRT|20|baseline\nL09|0|FedRPD|1|full\nL10|1|FedRPD|5|full\nL11|2|FedRPD|10|full\nL12|3|FedRPD|20|full\nL13|0|FedRPD|1|no_purify\nL14|1|FedRPD|5|no_purify\nL15|2|FedRPD|10|no_purify\nL16|3|FedRPD|20|no_purify\nL17|0|FedRPD|1|no_distill\nL18|1|FedRPD|5|no_distill\nL19|2|FedRPD|10|no_distill\nL20|3|FedRPD|20|no_distill\nL21|0|FedRPD|1|purify2\nL22|1|FedRPD|5|purify2\nL23|2|FedRPD|10|purify2\nL24|3|FedRPD|20|purify2'

echo "PFEDBA_ROOT=${PFEDBA_ROOT}" | tee "${LOGDIR}/launcher.env"
echo "RUN_TS=${RUN_TS}" | tee -a "${LOGDIR}/launcher.env"
echo "LOGDIR=${LOGDIR}" | tee -a "${LOGDIR}/launcher.env"
echo "PYTHON_BIN=${PYTHON_BIN}" | tee -a "${LOGDIR}/launcher.env"
echo -e "tag\tgpu\talgorithm\tlocal_epochs\tvariant" > "${LOGDIR}/manifest.tsv"
printf '%s\n' "${MANIFEST}" | tr '|' '\t' >> "${LOGDIR}/manifest.tsv"

build_cmd() {
  local algo="$1"
  local local_epochs="$2"
  local variant="$3"

  local -a cmd=(
    "${PYTHON_BIN}" -u "${PFEDBA_ROOT}/main.py"
    "${COMMON_ARGS[@]}"
    --algorithm "${algo}"
    --local_epochs "${local_epochs}"
  )

  if [[ "${algo}" == "FedRT" ]]; then
    cmd+=("${FEDRT_ARGS[@]}")
  elif [[ "${algo}" == "FedRPD" ]]; then
    cmd+=("${FEDRPD_ARGS[@]}")
    case "${variant}" in
      no_purify)
        cmd+=(--purify_beta 0)
        ;;
      no_distill)
        cmd+=(--distill_weight 0)
        ;;
      purify2)
        cmd+=(--purify_rounds 2)
        ;;
    esac
  fi

  printf '%q ' "${cmd[@]}"
}

launch_one() {
  local tag="$1"
  local gpu="$2"
  local algo="$3"
  local local_epochs="$4"
  local variant="$5"
  local log_file="${LOGDIR}/${tag}.log"
  local meta_file="${LOGDIR}/${tag}.meta.log"
  local cmd
  cmd="$(build_cmd "${algo}" "${local_epochs}" "${variant}")"

  (
    {
      echo "=================================================="
      echo "[START] ${tag}"
      echo "GPU=${gpu}"
      echo "ALGORITHM=${algo}"
      echo "LOCAL_EPOCHS=${local_epochs}"
      echo "VARIANT=${variant}"
      echo "TIME=$(date '+%Y-%m-%d %H:%M:%S')"
      echo "LOG=${log_file}"
      echo "CMD=${cmd}"
      echo "=================================================="
    } > "${meta_file}"

    CUDA_VISIBLE_DEVICES="${gpu}" bash -lc "${cmd}" > "${log_file}" 2>&1 < /dev/null
    rc=$?

    {
      echo "[END] ${tag} RC=${rc} TIME=$(date '+%Y-%m-%d %H:%M:%S')"
      echo "=================================================="
    } >> "${meta_file}"
    exit "${rc}"
  ) < /dev/null &
  LAUNCH_PID=$!
}

run_wave() {
  local wave_idx="$1"
  local start_line="$2"
  local end_line="$3"
  local pids=()

  echo "===== WAVE ${wave_idx}: lines ${start_line}-${end_line} =====" | tee -a "${LOGDIR}/waves.log"
  while IFS='|' read -r tag gpu algo local_epochs variant; do
    [[ -z "${tag}" ]] && continue
    launch_one "${tag}" "${gpu}" "${algo}" "${local_epochs}" "${variant}"
    pids+=("${LAUNCH_PID}")
  done < <(printf '%s\n' "${MANIFEST}" | sed -n "${start_line},${end_line}p")

  local fail=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      fail=1
    fi
  done
  return "${fail}"
}

overall_fail=0
run_wave 1 1 4 || overall_fail=1
run_wave 2 5 8 || overall_fail=1
run_wave 3 9 12 || overall_fail=1
run_wave 4 13 16 || overall_fail=1
run_wave 5 17 20 || overall_fail=1
run_wave 6 21 24 || overall_fail=1

{
  echo -e "tag\talgorithm\tlocal_epochs\tvariant\tpersonal_acc\tpersonal_asr\tglobal_acc\tglobal_asr"
  while IFS='|' read -r tag gpu algo local_epochs variant; do
    [[ -z "${tag}" ]] && continue
    log_file="${LOGDIR}/${tag}.log"
    global_acc=$(grep 'Evaluate the final global model' -A2 "${log_file}" | grep 'Average Global Accurancy' | tail -n1 | awk '{print $4}')
    global_asr=$(grep 'Evaluate the final global model' -A3 "${log_file}" | grep 'Average Global ATTACK ALL ASR' | tail -n1 | awk '{print $6}')
    personal_acc=$(grep 'Evaluate the final global model with a few step update' -A2 "${log_file}" | grep 'Average Personal Accurancy' | tail -n1 | awk '{print $6}')
    personal_asr=$(grep 'Evaluate the final global model with a few step update' -A3 "${log_file}" | grep 'Average Personal ATTACK ALL ASR' | tail -n1 | awk '{print $9}')
    echo -e "${tag}\t${algo}\t${local_epochs}\t${variant}\t${personal_acc}\t${personal_asr}\t${global_acc}\t${global_asr}"
  done <<< "${MANIFEST}"
} | tee "${LOGDIR}/summary.tsv"

if [[ "${overall_fail}" -ne 0 ]]; then
  echo "Some runs failed. Check ${LOGDIR}."
  exit 1
fi

echo "All 24 runs completed. Summary: ${LOGDIR}/summary.tsv"
