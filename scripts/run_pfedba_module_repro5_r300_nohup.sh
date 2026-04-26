#!/usr/bin/env bash
set -u

PFEDBA_ROOT="/home/huangtu/PFL_Backdoor_Defense/PFedBA"
PYTHON_BIN="${PYTHON_BIN:-/home/huangtu/miniconda3/envs/torch/bin/python}"
MAIN_PY="${PFEDBA_ROOT}/main.py"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${PFEDBA_ROOT}/log/module_repro5_r300_${RUN_TS}"
STATUS_DIR="${LOGDIR}/status"

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
  --num_global_iters 300
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

FEDRPD_FULL_ARGS=(
  --rt_beta 0.10
  --adv_eps 0.10
  --adv_num_iter 5
  --aug_strength 0.10
  --purify_beta 1500
  --purify_rounds 1
  --distill_gamma 1.0
  --distill_weight 1.0
)

FEDRPD_PURIFY_ONLY_ARGS=(
  --rt_beta 0.10
  --adv_eps 0.10
  --adv_num_iter 5
  --aug_strength 0.10
  --purify_beta 1500
  --purify_rounds 1
  --distill_gamma 1.0
  --distill_weight 0
)

FEDRPD_DISTILL_ONLY_ARGS=(
  --rt_beta 0.10
  --adv_eps 0.10
  --adv_num_iter 5
  --aug_strength 0.10
  --purify_beta 0
  --purify_rounds 1
  --distill_gamma 1.0
  --distill_weight 1.0
)

run_one() {
  local gpu="$1"
  local tag="$2"
  local algo="$3"
  local note="$4"
  shift 4

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
    echo "LOCAL_EPOCHS=10"
    echo "PLOCAL_EPOCHS=1"
    echo "GLOBAL_ITERS=300"
    echo "TIME=$(date '+%F %T')"
    echo "LOG=${log_file}"
    echo "=================================================="
  } >> "${meta_file}"

  env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" -u "${MAIN_PY}" \
    "${COMMON_ARGS[@]}" \
    --algorithm "${algo}" \
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
  run_one 0 M01 FedRep baseline
  run_one 0 M05 FedRPD full "${FEDRPD_FULL_ARGS[@]}"
}

queue_gpu1() {
  run_one 1 M02 FedRT fedrt_anchor "${FEDRT_ARGS[@]}"
}

queue_gpu2() {
  run_one 2 M03 FedRPD purify_only "${FEDRPD_PURIFY_ONLY_ARGS[@]}"
}

queue_gpu3() {
  run_one 3 M04 FedRPD distill_only "${FEDRPD_DISTILL_ONLY_ARGS[@]}"
}

cat > "${LOGDIR}/manifest.tsv" <<'EOF'
tag	gpu	algorithm	note	local_epochs	plocal_epochs	global_iters
M01	0	FedRep	baseline	10	1	300
M02	1	FedRT	fedrt_anchor	10	1	300
M03	2	FedRPD	purify_only	10	1	300
M04	3	FedRPD	distill_only	10	1	300
M05	0	FedRPD	full	10	1	300
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
  echo "GLOBAL_ITERS=300"
  echo "RT_BETA=0.10"
  echo "NOTE=with numusers=10, rt_beta=0.10 means trim k=1"
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
  echo -e "tag\talgorithm\tnote\tpersonal_acc\tpersonal_asr\tglobal_acc\tglobal_asr"
  for tag in M01 M02 M03 M04 M05; do
    log_file="${LOGDIR}/${tag}.log"
    global_acc=$(grep 'Evaluate the final global model' -A2 "${log_file}" | grep 'Average Global Accurancy' | tail -n1 | awk '{print $4}')
    global_asr=$(grep 'Evaluate the final global model' -A3 "${log_file}" | grep 'Average Global ATTACK ALL ASR' | tail -n1 | awk '{print $6}')
    personal_acc=$(grep 'Evaluate the final global model with a few step update' -A2 "${log_file}" | grep 'Average Personal Accurancy' | tail -n1 | awk '{print $6}')
    personal_asr=$(grep 'Evaluate the final global model with a few step update' -A3 "${log_file}" | grep 'Average Personal ATTACK ALL ASR' | tail -n1 | awk '{print $9}')
    note=$(awk -F '\t' -v t="${tag}" '$1==t {print $4}' "${LOGDIR}/manifest.tsv")
    algo=$(awk -F '\t' -v t="${tag}" '$1==t {print $3}' "${LOGDIR}/manifest.tsv")
    echo -e "${tag}\t${algo}\t${note}\t${personal_acc}\t${personal_asr}\t${global_acc}\t${global_asr}"
  done
} | tee "${LOGDIR}/summary.tsv"
