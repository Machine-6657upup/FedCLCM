#!/usr/bin/env bash
set -euo pipefail

PFEDBA_ROOT="${PFEDBA_ROOT:-/home/huangtu/PFL_Backdoor_Defense/PFedBA}"
PYTHON_BIN="${PYTHON_BIN:-/home/huangtu/miniconda3/envs/torch/bin/python}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-0}"
TAG="${TAG:-BDPFL_FULL}"
LOGDIR="${LOGDIR:-${PFEDBA_ROOT}/log/bdpfl_full_r300_${RUN_TS}}"
LOG_FILE="${LOGDIR}/${TAG}.log"
META_FILE="${LOGDIR}/${TAG}.meta.log"
SUMMARY_FILE="${LOGDIR}/summary.tsv"
LE="${LE:-20}"
GI="${GI:-300}"
BL="${BL:-1.0}"
BT="${BT:-1.0}"
BG="${BG:-1.0}"
BI="${BI:-1}"
BE="${BE:-1}"

mkdir -p "${LOGDIR}"
cd "${PFEDBA_ROOT}"

COMMON_ARGS=(
  --dataset Cifar10
  --model resnet
  --resnet_pretrained 0
  --algorithm BDPFL
  --learning_rate 0.1
  --local_epochs "${LE}"
  --num_global_iters "${GI}"
  --numusers 10
  --batch_size 64
  --attack_start 30
  --attack_method attackall
  --poisoning_per_batch 5
  --defense none
  --per_epoch 1
  --malclient 10
  --times 1
  --bd_lambda "${BL}"
  --bd_tau "${BT}"
  --bd_gamma "${BG}"
  --bd_use_inter "${BI}"
  --bd_use_em "${BE}"
)

{
  echo "PFEDBA_ROOT=${PFEDBA_ROOT}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
  echo "RUN_TS=${RUN_TS}"
  echo "GPU=${GPU}"
  echo "LOGDIR=${LOGDIR}"
  echo "TAG=${TAG}"
  echo "LOCAL_EPOCHS=${LE}"
  echo "GLOBAL_ITERS=${GI}"
  echo "BD_LAMBDA=${BL}"
  echo "BD_TAU=${BT}"
  echo "BD_GAMMA=${BG}"
  echo "BD_USE_INTER=${BI}"
  echo "BD_USE_EM=${BE}"
} | tee "${LOGDIR}/launcher.env"

{
  echo "=================================================="
  echo "[START] ${TAG}"
  echo "TIME=$(date '+%F %T')"
  echo "GPU=${GPU}"
  echo "LOG=${LOG_FILE}"
  printf 'CMD=%q ' "${PYTHON_BIN}" -u "${PFEDBA_ROOT}/main.py" "${COMMON_ARGS[@]}"
  echo
  echo "=================================================="
} | tee "${META_FILE}"

env CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl \
  "${PYTHON_BIN}" -u "${PFEDBA_ROOT}/main.py" \
  "${COMMON_ARGS[@]}" > "${LOG_FILE}" 2>&1

global_acc=$(grep 'Average Global Accurancy' "${LOG_FILE}" | tail -n1 | awk '{print $4}')
global_asr=$(grep 'Average Global ATTACK ALL ASR' "${LOG_FILE}" | tail -n1 | awk '{print $6}')
legacy_personal_acc=$(grep 'Average Personal Accurancy (k local SGD)' "${LOG_FILE}" | tail -n1 | awk '{print $6}')
legacy_personal_asr=$(grep 'Average Personal ATTACK ALL ASR (k local SGD)' "${LOG_FILE}" | tail -n1 | awk '{print $9}')
saved_personal_acc=$(grep '^Average Personal Accurancy:' "${LOG_FILE}" | tail -n1 | awk '{print $4}')
saved_personal_asr=$(grep 'Average Personal ATTACK ALL ASR (saved personalized model)' "${LOG_FILE}" | tail -n1 | awk '{print $8}')
saved_benign_acc=$(grep 'Average Personal Accurancy (saved personalized benign-only)' "${LOG_FILE}" | tail -n1 | awk -F': ' '{print $2}')
saved_benign_asr=$(grep 'Average Personal ATTACK ALL ASR (saved personalized benign-only)' "${LOG_FILE}" | tail -n1 | awk -F': ' '{print $2}')
saved_malicious_acc=$(grep 'Average Personal Accurancy (saved personalized malicious-only)' "${LOG_FILE}" | tail -n1 | awk -F': ' '{print $2}')
saved_malicious_asr=$(grep 'Average Personal ATTACK ALL ASR (saved personalized malicious-only)' "${LOG_FILE}" | tail -n1 | awk -F': ' '{print $2}')

{
  echo -e "tag\tglobal_acc\tglobal_asr\tlegacy_personal_acc\tlegacy_personal_asr\tsaved_personal_acc\tsaved_personal_asr\tsaved_benign_acc\tsaved_benign_asr\tsaved_malicious_acc\tsaved_malicious_asr"
  echo -e "${TAG}\t${global_acc}\t${global_asr}\t${legacy_personal_acc}\t${legacy_personal_asr}\t${saved_personal_acc}\t${saved_personal_asr}\t${saved_benign_acc}\t${saved_benign_asr}\t${saved_malicious_acc}\t${saved_malicious_asr}"
} | tee "${SUMMARY_FILE}"

{
  echo "[END] ${TAG} RC=0 TIME=$(date '+%F %T')"
  echo "SUMMARY=${SUMMARY_FILE}"
  echo "=================================================="
} | tee -a "${META_FILE}"
