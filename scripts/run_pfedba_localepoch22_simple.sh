#!/usr/bin/env bash
set -u

PFEDBA_ROOT="/home/huangtu/PFL_Backdoor_Defense/PFedBA"
PYTHON_BIN="${PYTHON_BIN:-/home/huangtu/miniconda3/envs/torch/bin/python}"
MAIN_PY="${PFEDBA_ROOT}/main.py"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-${PFEDBA_ROOT}/log/localepoch22_${RUN_TS}}"
STATUS_DIR="${LOGDIR}/status"

mkdir -p "${LOGDIR}" "${STATUS_DIR}"
cd "${PFEDBA_ROOT}" || exit 1

common_args=(
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

fedrt_args=(
  --rt_beta 0.10
  --adv_eps 0.10
  --adv_num_iter 5
  --aug_strength 0.10
)

fedrpd_args=(
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
    extra_args=("${fedrt_args[@]}")
  elif [[ "${algo}" == "FedRPD" ]]; then
    extra_args=("${fedrpd_args[@]}")
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
    "${common_args[@]}" \
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
}

write_manifest() {
  cat > "${LOGDIR}/manifest.tsv" <<'EOF'
tag	gpu	algorithm	local_epochs	variant
E01	0	FedRep	1	baseline
E02	1	FedRep	5	baseline
E03	2	FedRep	10	baseline
E04	3	FedRT	1	baseline
E05	0	FedRT	5	baseline
E06	1	FedRT	10	baseline
E07	2	FedRPD	1	full
E08	3	FedRPD	5	full
E09	0	FedRPD	10	full
E10	1	FedRPD	20	full
E11	2	FedRPD	1	no_purify
E12	3	FedRPD	5	no_purify
E13	0	FedRPD	10	no_purify
E14	1	FedRPD	20	no_purify
E15	2	FedRPD	1	no_distill
E16	3	FedRPD	5	no_distill
E17	0	FedRPD	10	no_distill
E18	1	FedRPD	20	no_distill
E19	2	FedRPD	1	purify2
E20	3	FedRPD	5	purify2
E21	0	FedRPD	10	purify2
E22	1	FedRPD	20	purify2
EOF
}

queue_gpu0() {
  run_one 0 E01 FedRep 1 baseline
  run_one 0 E05 FedRT 5 baseline
  run_one 0 E09 FedRPD 10 full
  run_one 0 E13 FedRPD 10 no_purify
  run_one 0 E17 FedRPD 10 no_distill
  run_one 0 E21 FedRPD 10 purify2
}

queue_gpu1() {
  run_one 1 E02 FedRep 5 baseline
  run_one 1 E06 FedRT 10 baseline
  run_one 1 E10 FedRPD 20 full
  run_one 1 E14 FedRPD 20 no_purify
  run_one 1 E18 FedRPD 20 no_distill
  run_one 1 E22 FedRPD 20 purify2
}

queue_gpu2() {
  run_one 2 E03 FedRep 10 baseline
  run_one 2 E07 FedRPD 1 full
  run_one 2 E11 FedRPD 1 no_purify
  run_one 2 E15 FedRPD 1 no_distill
  run_one 2 E19 FedRPD 1 purify2
}

queue_gpu3() {
  run_one 3 E04 FedRT 1 baseline
  run_one 3 E08 FedRPD 5 full
  run_one 3 E12 FedRPD 5 no_purify
  run_one 3 E16 FedRPD 5 no_distill
  run_one 3 E20 FedRPD 5 purify2
}

collect_summary() {
  {
    echo -e "tag\talgorithm\tlocal_epochs\tvariant\tpersonal_acc\tpersonal_asr\tglobal_acc\tglobal_asr"
    while IFS=$'\t' read -r tag gpu algo local_epochs variant; do
      [[ "${tag}" == "tag" ]] && continue
      local log_file="${LOGDIR}/${tag}.log"
      local global_acc
      local global_asr
      local personal_acc
      local personal_asr
      global_acc=$(grep 'Evaluate the final global model' -A2 "${log_file}" | grep 'Average Global Accurancy' | tail -n1 | awk '{print $4}')
      global_asr=$(grep 'Evaluate the final global model' -A3 "${log_file}" | grep 'Average Global ATTACK ALL ASR' | tail -n1 | awk '{print $6}')
      personal_acc=$(grep 'Evaluate the final global model with a few step update' -A2 "${log_file}" | grep 'Average Personal Accurancy' | tail -n1 | awk '{print $6}')
      personal_asr=$(grep 'Evaluate the final global model with a few step update' -A3 "${log_file}" | grep 'Average Personal ATTACK ALL ASR' | tail -n1 | awk '{print $9}')
      echo -e "${tag}\t${algo}\t${local_epochs}\t${variant}\t${personal_acc}\t${personal_asr}\t${global_acc}\t${global_asr}"
    done < "${LOGDIR}/manifest.tsv"
  } | tee "${LOGDIR}/summary.tsv"
}

echo "PFEDBA_ROOT=${PFEDBA_ROOT}" | tee "${LOGDIR}/launcher.env"
echo "RUN_TS=${RUN_TS}" | tee -a "${LOGDIR}/launcher.env"
echo "LOGDIR=${LOGDIR}" | tee -a "${LOGDIR}/launcher.env"
echo "STATUS_DIR=${STATUS_DIR}" | tee -a "${LOGDIR}/launcher.env"
echo "PYTHON_BIN=${PYTHON_BIN}" | tee -a "${LOGDIR}/launcher.env"
echo "MAIN_PY=${MAIN_PY}" | tee -a "${LOGDIR}/launcher.env"
echo "REUSE_BASELINES=FedRep(local_epochs=20),FedRT(local_epochs=20)" | tee -a "${LOGDIR}/launcher.env"

write_manifest

queue_gpu0 &
PID0=$!
queue_gpu1 &
PID1=$!
queue_gpu2 &
PID2=$!
queue_gpu3 &
PID3=$!

rc=0
for pid in "${PID0}" "${PID1}" "${PID2}" "${PID3}"; do
  if ! wait "${pid}"; then
    rc=1
  fi
done

collect_summary
exit "${rc}"
