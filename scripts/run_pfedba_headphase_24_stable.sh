#!/usr/bin/env bash
set -euo pipefail

PFEDBA_ROOT="${PFEDBA_ROOT:-/home/huangtu/PFL_Backdoor_Defense/PFedBA}"
PYTHON_BIN="${PYTHON_BIN:-/home/huangtu/miniconda3/envs/torch/bin/python}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-${PFEDBA_ROOT}/log/headphase24_${RUN_TS}}"

mkdir -p "${LOGDIR}"

COMMON_ARGS=(
  --dataset Cifar10
  --model resnet
  --resnet_pretrained 0
  --learning_rate 0.1
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

FEDRT_EXTRA=(
  --rt_beta 0.10
  --adv_eps 0.10
  --adv_num_iter 5
  --aug_strength 0.10
)

MANIFEST=$'R01|0|FedRep|1|0.10\nR02|1|FedRep|2|0.10\nR03|2|FedRep|5|0.10\nR04|3|FedRep|10|0.10\nR05|0|FedRep|1|0.05\nR06|1|FedRep|2|0.05\nR07|2|FedRep|5|0.05\nR08|3|FedRep|10|0.05\nR09|0|FedRep|1|0.02\nR10|1|FedRep|2|0.02\nR11|2|FedRep|5|0.02\nR12|3|FedRep|10|0.02\nT01|0|FedRT|1|0.10\nT02|1|FedRT|2|0.10\nT03|2|FedRT|5|0.10\nT04|3|FedRT|10|0.10\nT05|0|FedRT|1|0.05\nT06|1|FedRT|2|0.05\nT07|2|FedRT|5|0.05\nT08|3|FedRT|10|0.05\nT09|0|FedRT|1|0.02\nT10|1|FedRT|2|0.02\nT11|2|FedRT|5|0.02\nT12|3|FedRT|10|0.02'

echo "PFEDBA_ROOT=${PFEDBA_ROOT}" | tee "${LOGDIR}/launcher.env"
echo "RUN_TS=${RUN_TS}" | tee -a "${LOGDIR}/launcher.env"
echo "LOGDIR=${LOGDIR}" | tee -a "${LOGDIR}/launcher.env"
echo "PYTHON_BIN=${PYTHON_BIN}" | tee -a "${LOGDIR}/launcher.env"

echo -e "tag\tgpu\talgorithm\tplocal_epochs\tlr_head" > "${LOGDIR}/manifest.tsv"
printf '%s\n' "${MANIFEST}" | tr '|' '\t' >> "${LOGDIR}/manifest.tsv"

launch_one() {
  local tag="$1"
  local gpu="$2"
  local algo="$3"
  local plocal="$4"
  local lr_head="$5"
  local log_file="${LOGDIR}/${tag}.log"
  local meta_file="${LOGDIR}/${tag}.meta.log"

  (
    {
      echo "=================================================="
      echo "[START] ${tag}"
      echo "GPU=${gpu}"
      echo "ALGORITHM=${algo}"
      echo "PLOCAL_EPOCHS=${plocal}"
      echo "LR_HEAD=${lr_head}"
      echo "TIME=$(date '+%Y-%m-%d %H:%M:%S')"
      echo "LOG=${log_file}"
      echo "=================================================="
    } > "${meta_file}"

    cmd=(
      "${PYTHON_BIN}" -u "${PFEDBA_ROOT}/main.py"
      "${COMMON_ARGS[@]}"
      --algorithm "${algo}"
      --plocal_epochs "${plocal}"
      --lr_head "${lr_head}"
    )
    if [[ "${algo}" == "FedRT" ]]; then
      cmd+=("${FEDRT_EXTRA[@]}")
    fi

    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" > "${log_file}" 2>&1
    rc=$?

    {
      echo "[END] ${tag} RC=${rc} TIME=$(date '+%Y-%m-%d %H:%M:%S')"
      echo "=================================================="
    } >> "${meta_file}"
    exit "${rc}"
  ) &
  echo $!
}

run_wave() {
  local wave_idx="$1"
  local start_line="$2"
  local end_line="$3"
  local pids=()

  echo "===== WAVE ${wave_idx}: lines ${start_line}-${end_line} =====" | tee -a "${LOGDIR}/waves.log"
  while IFS='|' read -r tag gpu algo plocal lr_head; do
    [[ -z "${tag}" ]] && continue
    pid=$(launch_one "${tag}" "${gpu}" "${algo}" "${plocal}" "${lr_head}")
    pids+=("${pid}")
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
  echo -e "tag\talgorithm\tplocal_epochs\tlr_head\tpersonal_acc\tpersonal_asr\tglobal_acc\tglobal_asr"
  while IFS='|' read -r tag gpu algo plocal lr_head; do
    [[ -z "${tag}" ]] && continue
    log_file="${LOGDIR}/${tag}.log"
    global_acc=$(grep 'Evaluate the final global model' -A2 "${log_file}" | grep 'Average Global Accurancy' | tail -n1 | awk '{print $4}')
    global_asr=$(grep 'Evaluate the final global model' -A3 "${log_file}" | grep 'Average Global ATTACK ALL ASR' | tail -n1 | awk '{print $6}')
    personal_acc=$(grep 'Evaluate the final global model with a few step update' -A2 "${log_file}" | grep 'Average Personal Accurancy' | tail -n1 | awk '{print $6}')
    personal_asr=$(grep 'Evaluate the final global model with a few step update' -A3 "${log_file}" | grep 'Average Personal ATTACK ALL ASR' | tail -n1 | awk '{print $9}')
    echo -e "${tag}\t${algo}\t${plocal}\t${lr_head}\t${personal_acc}\t${personal_asr}\t${global_acc}\t${global_asr}"
  done <<< "${MANIFEST}"
} | tee "${LOGDIR}/summary.tsv"

if [[ "${overall_fail}" -ne 0 ]]; then
  echo "Some runs failed. Check ${LOGDIR}."
  exit 1
fi

echo "All 24 runs completed. Summary: ${LOGDIR}/summary.tsv"
