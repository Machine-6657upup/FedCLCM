#!/usr/bin/env bash
set -euo pipefail

PFEDBA_ROOT="${PFEDBA_ROOT:-/home/huangtu/PFL_Backdoor_Defense/PFedBA}"
PYTHON_BIN="${PYTHON_BIN:-/home/huangtu/miniconda3/envs/torch/bin/python}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-${PFEDBA_ROOT}/log/fedrep_head_epoch_${RUN_TS}}"

mkdir -p "${LOGDIR}"

PLOCAL_LIST=(1 2 5 10)
GPUS=(0 1 2 3)

COMMON_ARGS=(
  --dataset Cifar10
  --model resnet
  --resnet_pretrained 0
  --algorithm FedRep
  --learning_rate 0.1
  --lr_head 0.1
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

echo "PFEDBA_ROOT=${PFEDBA_ROOT}" | tee "${LOGDIR}/launcher.env"
echo "RUN_TS=${RUN_TS}" | tee -a "${LOGDIR}/launcher.env"
echo "LOGDIR=${LOGDIR}" | tee -a "${LOGDIR}/launcher.env"
echo "PYTHON_BIN=${PYTHON_BIN}" | tee -a "${LOGDIR}/launcher.env"

declare -a PIDS=()
declare -a TAGS=()

launch_one() {
  local tag="$1"
  local gpu="$2"
  local plocal="$3"
  local log_file="${LOGDIR}/${tag}_plocal${plocal}.log"
  local meta_file="${LOGDIR}/${tag}.meta.log"

  (
    {
      echo "=================================================="
      echo "[START] ${tag}"
      echo "GPU=${gpu}"
      echo "PLOCAL_EPOCHS=${plocal}"
      echo "TIME=$(date '+%Y-%m-%d %H:%M:%S')"
      echo "LOG=${log_file}"
      echo "=================================================="
    } > "${meta_file}"

    CUDA_VISIBLE_DEVICES="${gpu}" \
      "${PYTHON_BIN}" -u "${PFEDBA_ROOT}/main.py" \
      "${COMMON_ARGS[@]}" \
      --plocal_epochs "${plocal}" \
      > "${log_file}" 2>&1
    rc=$?

    {
      echo "[END] ${tag} RC=${rc} TIME=$(date '+%Y-%m-%d %H:%M:%S')"
      echo "=================================================="
    } >> "${meta_file}"
    exit "${rc}"
  ) &

  PIDS+=("$!")
  TAGS+=("${tag}")
}

for idx in "${!PLOCAL_LIST[@]}"; do
  tag=$(printf "H%02d" "$((idx + 1))")
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  plocal="${PLOCAL_LIST[$idx]}"
  launch_one "${tag}" "${gpu}" "${plocal}"
done

fail=0
for idx in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$idx]}"; then
    fail=1
  fi
done

{
  echo -e "tag\tplocal_epochs\tpersonal_acc\tpersonal_asr\tglobal_acc\tglobal_asr"
  for idx in "${!PLOCAL_LIST[@]}"; do
    tag=$(printf "H%02d" "$((idx + 1))")
    plocal="${PLOCAL_LIST[$idx]}"
    log_file="${LOGDIR}/${tag}_plocal${plocal}.log"
    global_acc=$(grep 'Evaluate the final global model' -A2 "${log_file}" | grep 'Average Global Accurancy' | tail -n1 | awk '{print $4}')
    global_asr=$(grep 'Evaluate the final global model' -A3 "${log_file}" | grep 'Average Global ATTACK ALL ASR' | tail -n1 | awk '{print $6}')
    personal_acc=$(grep 'Evaluate the final global model with a few step update' -A2 "${log_file}" | grep 'Average Personal Accurancy' | tail -n1 | awk '{print $6}')
    personal_asr=$(grep 'Evaluate the final global model with a few step update' -A3 "${log_file}" | grep 'Average Personal ATTACK ALL ASR' | tail -n1 | awk '{print $9}')
    echo -e "${tag}\t${plocal}\t${personal_acc}\t${personal_asr}\t${global_acc}\t${global_asr}"
  done
} | tee "${LOGDIR}/summary.tsv"

if [[ "${fail}" -ne 0 ]]; then
  echo "Some runs failed. Check ${LOGDIR}."
  exit 1
fi

echo "All runs completed. Summary: ${LOGDIR}/summary.tsv"
