#!/usr/bin/env bash
set -euo pipefail

PFEDBA_ROOT="${PFEDBA_ROOT:-/home/huangtu/PFL_Backdoor_Defense/PFedBA}"
PYTHON_BIN="${PYTHON_BIN:-/home/huangtu/miniconda3/envs/torch/bin/python}"
MAIN_PY="${PFEDBA_ROOT}/main.py"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-${PFEDBA_ROOT}/log/pfedba_8_fullmix_${RUN_TS}}"
STATUS_DIR="${LOGDIR}/status"

mkdir -p "${LOGDIR}" "${STATUS_DIR}"
cd "${PFEDBA_ROOT}"

COMMON_ARGS=(
  --dataset Cifar10
  --model resnet
  --resnet_pretrained 0
  --learning_rate 0.1
  --lr_head 0.1
  --plocal_epochs 1
  --local_epochs 10
  --num_global_iters 400
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

LITE_BASE_ARGS=(
  --algorithm FedRPD
  --rt_beta 0
  --adv_eps 0
  --adv_num_iter 0
  --aug_strength 0
  --distill_gamma 1.0
)

run_one() {
  local gpu="$1"
  local tag="$2"
  local algo="$3"
  local note="$4"
  local metric_kind="$5"
  shift 5

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
    echo "ALGORITHM=${algo}"
    echo "NOTE=${note}"
    echo "METRIC_KIND=${metric_kind}"
    echo "TIME=$(date '+%F %T')"
    echo "LOG=${log_file}"
    printf 'CMD=%q ' "${PYTHON_BIN}" -u "${MAIN_PY}" "${COMMON_ARGS[@]}" "$@"
    echo
    echo "=================================================="
  } | tee "${meta_file}"

  env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp PYTHONPYCACHEPREFIX=/tmp \
    "${PYTHON_BIN}" -u "${MAIN_PY}" \
    "${COMMON_ARGS[@]}" \
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
  } | tee -a "${meta_file}"
}

queue_gpu0() {
  run_one 0 X01 FedRep baseline legacy_personal --algorithm FedRep
  run_one 0 X05 PFLALP full_default saved_benign --algorithm PFLALP --purify_beta 1500 --purify_rounds 1 --cluster_max_k 4
}

queue_gpu1() {
  run_one 1 X02 FedRPD fedrep_plus_purify_only legacy_personal \
    "${LITE_BASE_ARGS[@]}" --purify_beta 1500 --purify_rounds 1 --distill_weight 0
  run_one 1 X06 PFLALP full_pr2 saved_benign --algorithm PFLALP --purify_beta 1500 --purify_rounds 2 --cluster_max_k 4
}

queue_gpu2() {
  run_one 2 X03 FedRPD fedrep_plus_distill_only legacy_personal \
    "${LITE_BASE_ARGS[@]}" --purify_beta 0 --purify_rounds 1 --distill_weight 1.0
  run_one 2 X07 BDPFL full_default saved_benign --algorithm BDPFL --bd_lambda 1.0 --bd_tau 1.0 --bd_gamma 1.0 --bd_use_inter 1 --bd_use_em 1
}

queue_gpu3() {
  run_one 3 X04 FedRPD fedrep_plus_purify_distill legacy_personal \
    "${LITE_BASE_ARGS[@]}" --purify_beta 1500 --purify_rounds 1 --distill_weight 1.0
  run_one 3 X08 BDPFL full_gamma2 saved_benign --algorithm BDPFL --bd_lambda 1.0 --bd_tau 1.0 --bd_gamma 2.0 --bd_use_inter 1 --bd_use_em 1
}

cat > "${LOGDIR}/manifest.tsv" <<'EOF'
tag	gpu	algorithm	note	metric_kind
X01	0	FedRep	baseline	legacy_personal
X02	1	FedRPD	fedrep_plus_purify_only	legacy_personal
X03	2	FedRPD	fedrep_plus_distill_only	legacy_personal
X04	3	FedRPD	fedrep_plus_purify_distill	legacy_personal
X05	0	PFLALP	full_default	saved_benign
X06	1	PFLALP	full_pr2	saved_benign
X07	2	BDPFL	full_default	saved_benign
X08	3	BDPFL	full_gamma2	saved_benign
EOF

{
  echo "PFEDBA_ROOT=${PFEDBA_ROOT}"
  echo "RUN_TS=${RUN_TS}"
  echo "LOGDIR=${LOGDIR}"
  echo "STATUS_DIR=${STATUS_DIR}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
  echo "MAIN_PY=${MAIN_PY}"
  echo "LOCAL_EPOCHS=10"
  echo "PLOCAL_EPOCHS=1"
  echo "GLOBAL_ITERS=400"
  echo "NOTE=8-run batch: FedRep baseline, clean lite components, and two full methods"
} | tee "${LOGDIR}/launcher.env"

queue_gpu0 &
PID0=$!
queue_gpu1 &
PID1=$!
queue_gpu2 &
PID2=$!
queue_gpu3 &
PID3=$!

wait "${PID0}" "${PID1}" "${PID2}" "${PID3}"

{
  echo -e "tag\talgorithm\tnote\tmetric_kind\treport_acc\treport_asr\tglobal_acc\tglobal_asr\tlegacy_personal_acc\tlegacy_personal_asr\tsaved_personal_acc\tsaved_personal_asr\tsaved_benign_acc\tsaved_benign_asr"
  for tag in X01 X02 X03 X04 X05 X06 X07 X08; do
    log_file="${LOGDIR}/${tag}.log"
    note=$(awk -F '\t' -v t="${tag}" '$1==t {print $4}' "${LOGDIR}/manifest.tsv")
    algo=$(awk -F '\t' -v t="${tag}" '$1==t {print $3}' "${LOGDIR}/manifest.tsv")
    metric_kind=$(awk -F '\t' -v t="${tag}" '$1==t {print $5}' "${LOGDIR}/manifest.tsv")

    global_acc=$(grep 'Average Global Accurancy' "${log_file}" | tail -n1 | awk '{print $4}')
    global_asr=$(grep 'Average Global ATTACK ALL ASR' "${log_file}" | tail -n1 | awk '{print $6}')
    legacy_personal_acc=$(grep 'Average Personal Accurancy (k local SGD)' "${log_file}" | tail -n1 | awk '{print $6}')
    legacy_personal_asr=$(grep 'Average Personal ATTACK ALL ASR (k local SGD)' "${log_file}" | tail -n1 | awk '{print $9}')
    saved_personal_acc=$(grep '^Average Personal Accurancy:' "${log_file}" | tail -n1 | awk '{print $4}')
    saved_personal_asr=$(grep 'Average Personal ATTACK ALL ASR (saved personalized model)' "${log_file}" | tail -n1 | awk '{print $8}')
    saved_benign_acc=$(grep 'Average Personal Accurancy (saved personalized benign-only)' "${log_file}" | tail -n1 | awk -F': ' '{print $2}')
    saved_benign_asr=$(grep 'Average Personal ATTACK ALL ASR (saved personalized benign-only)' "${log_file}" | tail -n1 | awk -F': ' '{print $2}')

    report_acc="${legacy_personal_acc}"
    report_asr="${legacy_personal_asr}"
    if [[ "${metric_kind}" == "saved_benign" ]]; then
      report_acc="${saved_benign_acc}"
      report_asr="${saved_benign_asr}"
    fi

    echo -e "${tag}\t${algo}\t${note}\t${metric_kind}\t${report_acc}\t${report_asr}\t${global_acc}\t${global_asr}\t${legacy_personal_acc}\t${legacy_personal_asr}\t${saved_personal_acc}\t${saved_personal_asr}\t${saved_benign_acc}\t${saved_benign_asr}"
  done
} | tee "${LOGDIR}/summary.tsv"
