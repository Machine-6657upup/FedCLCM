#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC="${ROOT}/src"
FORMAL_DIR="${ROOT}/thesis_formal_logs"
PY="${PYTHON_BIN:-/home/huangtu/miniconda3/envs/torch/bin/python}"
WRAPPER="${ROOT}/scripts/generate_dataset_via_existing.py"
GEN_DIR="${SRC}/dataset/utils"
RAWDATA="${SRC}/dataset/rawdata"

MODE="${MODE:-gate}"   # gate | main
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-${RUN_TS}_static_thesis_effective_${MODE}}"
RUN_ROOT="${FORMAL_DIR}/${RUN_NAME}"
DATASET_LOG_DIR="${RUN_ROOT}/dataset_logs"
QUEUE_DIR="${RUN_ROOT}/queue_scripts"
LAUNCHER_LOG_DIR="${RUN_ROOT}/launcher_logs"
TRAIN_LOG_DIR="${RUN_ROOT}/train_logs"
PID_DIR="${RUN_ROOT}/pids"

NUM_CLIENTS="${NUM_CLIENTS:-100}"
JOIN_RATIO="${JOIN_RATIO:-0.1}"
LOCAL_LR="${LOCAL_LR:-0.1}"
LR_HEAD="${LR_HEAD:-0.1}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
PLOCAL_EPOCHS="${PLOCAL_EPOCHS:-1}"
ADV_CLIENTS="${ADV_CLIENTS:-10}"
BACKDOOR_RATE="${BACKDOOR_RATE:-0.2}"
GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-600}"
EVAL_GAP="${EVAL_GAP:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BASE_MODEL="${BASE_MODEL:-ResNet18}"

SIG_DELTA="${SIG_DELTA:-0.11764705882352941}" # 30/255
SIG_F="${SIG_F:-6}"
BLEND_ALPHA="${BLEND_ALPHA:-0.2}"

BADNET_DATASET="Cifar10_dir0.5_bdoor${BACKDOOR_RATE}_nclient_${NUM_CLIENTS}_badnet_adv${ADV_CLIENTS}"
BLEND_DATASET="Cifar10_dir0.5_bdoor${BACKDOOR_RATE}_nclient_${NUM_CLIENTS}_blend_adv${ADV_CLIENTS}"
SIG_DATASET="Cifar10_dir0.5_bdoor${BACKDOOR_RATE}_nclient_${NUM_CLIENTS}_sig_adv${ADV_CLIENTS}"

mkdir -p "${FORMAL_DIR}" "${DATASET_LOG_DIR}" "${QUEUE_DIR}" "${LAUNCHER_LOG_DIR}" "${TRAIN_LOG_DIR}" "${PID_DIR}"

ensure_mode() {
  if [[ "${MODE}" != "gate" && "${MODE}" != "main" ]]; then
    echo "[FATAL] MODE must be gate or main, got: ${MODE}" >&2
    exit 1
  fi
}

ensure_gpus() {
  local count
  count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
  if [[ -z "${count}" || "${count}" -lt 4 ]]; then
    echo "[FATAL] this launcher expects at least 4 visible GPUs." >&2
    exit 1
  fi
}

gen_dataset_if_missing() {
  local generator="$1"
  local dataset_name="$2"
  local log_name="$3"
  shift 3
  local config_path="${SRC}/dataset/${dataset_name}/config.json"

  if [[ -f "${config_path}" ]]; then
    echo "[DATASET] exists ${dataset_name}"
    return 0
  fi

  echo "[DATASET] generating ${dataset_name}"
  "${PY}" "${WRAPPER}" \
    --generator "${GEN_DIR}/${generator}" \
    --dir-path "${SRC}/dataset/${dataset_name}" \
    --rawdata-path "${RAWDATA}" \
    --num-clients "${NUM_CLIENTS}" \
    --backdoor-rate "${BACKDOOR_RATE}" \
    --adversary-num "${ADV_CLIENTS}" \
    --target-y 0 \
    --alpha 0.5 \
    --train-ratio 0.8 \
    --batch-size 10 \
    --partition dir \
    --niid \
    --balance \
    "$@" \
    > "${DATASET_LOG_DIR}/${log_name}" 2>&1
}

launch_queue() {
  local queue_name="$1"
  local script_path="${QUEUE_DIR}/${queue_name}.sh"
  local master_log="${LAUNCHER_LOG_DIR}/${queue_name}.master.log"

  nohup bash "${script_path}" > "${master_log}" 2>&1 < /dev/null &
  echo "$!" > "${PID_DIR}/${queue_name}.pid"
  echo "[LAUNCHED] ${queue_name} pid=$(cat "${PID_DIR}/${queue_name}.pid")"
}

write_readme() {
  cat > "${RUN_ROOT}/README.txt" <<EOF
Static thesis-effective run
Timestamp: ${RUN_TS}
Mode: ${MODE}

Purpose:
- lock the static attack setting to the already validated FedRep-break regime
- use a gate stage so we do not waste time on a weak attack setting
- rerun the main static table with clean, separated formal logs

Locked attack-effective regime:
- dataset family = Cifar10_dir0.5_bdoor${BACKDOOR_RATE}_nclient_${NUM_CLIENTS}_{badnet,blend,sig}_adv${ADV_CLIENTS}
- num_clients = ${NUM_CLIENTS}
- join_ratio = ${JOIN_RATIO}
- local_lr = ${LOCAL_LR}
- lr_head = ${LR_HEAD}
- local_epochs = ${LOCAL_EPOCHS}
- plocal_epochs = ${PLOCAL_EPOCHS}
- batch_size = ${BATCH_SIZE}
- model = ${BASE_MODEL}
- num_adv_clients = ${ADV_CLIENTS}
- global_rounds = ${GLOBAL_ROUNDS}
- eval_gap = ${EVAL_GAP}

Attack-generation defaults:
- blend_alpha = ${BLEND_ALPHA}
- sig_delta = ${SIG_DELTA}
- sig_f = ${SIG_F}
- sig_label_mode = dirty

FedCLCM explicit args:
- rt_beta = 0.2
- lambda_cl = 0.20
- aug_strength = 0.1
- adv_eps = 0
- adv_num_iter = 0
- mask_tau = 12.0
- mask_alpha = 0.70

Gate criterion:
- if all three FedRep runs stay below ASR = 0.2, treat the attack setting as invalid and do not trust the main table
EOF
}

write_manifest_gate() {
  cat > "${RUN_ROOT}/manifest.tsv" <<EOF
task_tag	gpu	algorithm	dataset	model	join_ratio	lr	lr_head	local_epochs	plocal_epochs	rounds
gate_rep_badnet	0	FedRep	${BADNET_DATASET}	${BASE_MODEL}	${JOIN_RATIO}	${LOCAL_LR}	${LR_HEAD}	${LOCAL_EPOCHS}	${PLOCAL_EPOCHS}	${GLOBAL_ROUNDS}
gate_rep_blend	1	FedRep	${BLEND_DATASET}	${BASE_MODEL}	${JOIN_RATIO}	${LOCAL_LR}	${LR_HEAD}	${LOCAL_EPOCHS}	${PLOCAL_EPOCHS}	${GLOBAL_ROUNDS}
gate_rep_sig	2	FedRep	${SIG_DATASET}	${BASE_MODEL}	${JOIN_RATIO}	${LOCAL_LR}	${LR_HEAD}	${LOCAL_EPOCHS}	${PLOCAL_EPOCHS}	${GLOBAL_ROUNDS}
EOF
}

write_manifest_main() {
  cat > "${RUN_ROOT}/manifest.tsv" <<EOF
task_tag	gpu	algorithm	dataset	model	join_ratio	lr	lr_head	local_epochs	plocal_epochs	rounds
main_badnet_fedavg	0	FedAvg	${BADNET_DATASET}	${BASE_MODEL}	${JOIN_RATIO}	${LOCAL_LR}	-	${LOCAL_EPOCHS}	-	${GLOBAL_ROUNDS}
main_badnet_fedrep	0	FedRep	${BADNET_DATASET}	${BASE_MODEL}	${JOIN_RATIO}	${LOCAL_LR}	${LR_HEAD}	${LOCAL_EPOCHS}	${PLOCAL_EPOCHS}	${GLOBAL_ROUNDS}
main_badnet_fedtrimmed	0	FedTrimmed	${BADNET_DATASET}	${BASE_MODEL}	${JOIN_RATIO}	${LOCAL_LR}	-	${LOCAL_EPOCHS}	-	${GLOBAL_ROUNDS}
main_blend_fedavg	1	FedAvg	${BLEND_DATASET}	${BASE_MODEL}	${JOIN_RATIO}	${LOCAL_LR}	-	${LOCAL_EPOCHS}	-	${GLOBAL_ROUNDS}
main_blend_fedrep	1	FedRep	${BLEND_DATASET}	${BASE_MODEL}	${JOIN_RATIO}	${LOCAL_LR}	${LR_HEAD}	${LOCAL_EPOCHS}	${PLOCAL_EPOCHS}	${GLOBAL_ROUNDS}
main_blend_fedtrimmed	1	FedTrimmed	${BLEND_DATASET}	${BASE_MODEL}	${JOIN_RATIO}	${LOCAL_LR}	-	${LOCAL_EPOCHS}	-	${GLOBAL_ROUNDS}
main_sig_fedavg	2	FedAvg	${SIG_DATASET}	${BASE_MODEL}	${JOIN_RATIO}	${LOCAL_LR}	-	${LOCAL_EPOCHS}	-	${GLOBAL_ROUNDS}
main_sig_fedrep	2	FedRep	${SIG_DATASET}	${BASE_MODEL}	${JOIN_RATIO}	${LOCAL_LR}	${LR_HEAD}	${LOCAL_EPOCHS}	${PLOCAL_EPOCHS}	${GLOBAL_ROUNDS}
main_sig_fedtrimmed	2	FedTrimmed	${SIG_DATASET}	${BASE_MODEL}	${JOIN_RATIO}	${LOCAL_LR}	-	${LOCAL_EPOCHS}	-	${GLOBAL_ROUNDS}
main_badnet_fedclcm	3	FedCLCM	${BADNET_DATASET}	${BASE_MODEL}	${JOIN_RATIO}	${LOCAL_LR}	${LR_HEAD}	${LOCAL_EPOCHS}	${PLOCAL_EPOCHS}	${GLOBAL_ROUNDS}
main_blend_fedclcm	3	FedCLCM	${BLEND_DATASET}	${BASE_MODEL}	${JOIN_RATIO}	${LOCAL_LR}	${LR_HEAD}	${LOCAL_EPOCHS}	${PLOCAL_EPOCHS}	${GLOBAL_ROUNDS}
main_sig_fedclcm	3	FedCLCM	${SIG_DATASET}	${BASE_MODEL}	${JOIN_RATIO}	${LOCAL_LR}	${LR_HEAD}	${LOCAL_EPOCHS}	${PLOCAL_EPOCHS}	${GLOBAL_ROUNDS}
EOF
}

write_gate_queues() {
  cat > "${QUEUE_DIR}/gpu0_queue.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${SRC}"
"${PY}" -u main.py \
  -dev cuda -did 0 \
  -data "${BADNET_DATASET}" \
  -m "${BASE_MODEL}" -algo FedRep \
  -ncl 10 -nc "${NUM_CLIENTS}" -jr "${JOIN_RATIO}" -lbs "${BATCH_SIZE}" \
  -lr "${LOCAL_LR}" -lr_head "${LR_HEAD}" -ls "${LOCAL_EPOCHS}" -pls "${PLOCAL_EPOCHS}" \
  -gr "${GLOBAL_ROUNDS}" -eg "${EVAL_GAP}" \
  -go gate_rep_badnet \
  --num_adv_clients "${ADV_CLIENTS}" \
  > "${TRAIN_LOG_DIR}/gate_rep_badnet.log" 2>&1
EOF

  cat > "${QUEUE_DIR}/gpu1_queue.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${SRC}"
"${PY}" -u main.py \
  -dev cuda -did 1 \
  -data "${BLEND_DATASET}" \
  -m "${BASE_MODEL}" -algo FedRep \
  -ncl 10 -nc "${NUM_CLIENTS}" -jr "${JOIN_RATIO}" -lbs "${BATCH_SIZE}" \
  -lr "${LOCAL_LR}" -lr_head "${LR_HEAD}" -ls "${LOCAL_EPOCHS}" -pls "${PLOCAL_EPOCHS}" \
  -gr "${GLOBAL_ROUNDS}" -eg "${EVAL_GAP}" \
  -go gate_rep_blend \
  --num_adv_clients "${ADV_CLIENTS}" \
  > "${TRAIN_LOG_DIR}/gate_rep_blend.log" 2>&1
EOF

  cat > "${QUEUE_DIR}/gpu2_queue.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${SRC}"
"${PY}" -u main.py \
  -dev cuda -did 2 \
  -data "${SIG_DATASET}" \
  -m "${BASE_MODEL}" -algo FedRep \
  -ncl 10 -nc "${NUM_CLIENTS}" -jr "${JOIN_RATIO}" -lbs "${BATCH_SIZE}" \
  -lr "${LOCAL_LR}" -lr_head "${LR_HEAD}" -ls "${LOCAL_EPOCHS}" -pls "${PLOCAL_EPOCHS}" \
  -gr "${GLOBAL_ROUNDS}" -eg "${EVAL_GAP}" \
  -go gate_rep_sig \
  --num_adv_clients "${ADV_CLIENTS}" \
  > "${TRAIN_LOG_DIR}/gate_rep_sig.log" 2>&1
EOF

  chmod +x "${QUEUE_DIR}/gpu0_queue.sh" "${QUEUE_DIR}/gpu1_queue.sh" "${QUEUE_DIR}/gpu2_queue.sh"
}

write_main_queues() {
  cat > "${QUEUE_DIR}/gpu0_queue.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${SRC}"

"${PY}" -u main.py \
  -dev cuda -did 0 \
  -data "${BADNET_DATASET}" \
  -m "${BASE_MODEL}" -algo FedAvg \
  -ncl 10 -nc "${NUM_CLIENTS}" -jr "${JOIN_RATIO}" -lbs "${BATCH_SIZE}" \
  -lr "${LOCAL_LR}" -ls "${LOCAL_EPOCHS}" \
  -gr "${GLOBAL_ROUNDS}" -eg "${EVAL_GAP}" \
  -go main_badnet_fedavg \
  --num_adv_clients "${ADV_CLIENTS}" \
  > "${TRAIN_LOG_DIR}/main_badnet_fedavg.log" 2>&1

"${PY}" -u main.py \
  -dev cuda -did 0 \
  -data "${BADNET_DATASET}" \
  -m "${BASE_MODEL}" -algo FedRep \
  -ncl 10 -nc "${NUM_CLIENTS}" -jr "${JOIN_RATIO}" -lbs "${BATCH_SIZE}" \
  -lr "${LOCAL_LR}" -lr_head "${LR_HEAD}" -ls "${LOCAL_EPOCHS}" -pls "${PLOCAL_EPOCHS}" \
  -gr "${GLOBAL_ROUNDS}" -eg "${EVAL_GAP}" \
  -go main_badnet_fedrep \
  --num_adv_clients "${ADV_CLIENTS}" \
  > "${TRAIN_LOG_DIR}/main_badnet_fedrep.log" 2>&1

"${PY}" -u main.py \
  -dev cuda -did 0 \
  -data "${BADNET_DATASET}" \
  -m "${BASE_MODEL}" -algo FedTrimmed \
  -ncl 10 -nc "${NUM_CLIENTS}" -jr "${JOIN_RATIO}" -lbs "${BATCH_SIZE}" \
  -lr "${LOCAL_LR}" -ls "${LOCAL_EPOCHS}" \
  -gr "${GLOBAL_ROUNDS}" -eg "${EVAL_GAP}" \
  -go main_badnet_fedtrimmed \
  --num_adv_clients "${ADV_CLIENTS}" \
  > "${TRAIN_LOG_DIR}/main_badnet_fedtrimmed.log" 2>&1
EOF

  cat > "${QUEUE_DIR}/gpu1_queue.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${SRC}"

"${PY}" -u main.py \
  -dev cuda -did 1 \
  -data "${BLEND_DATASET}" \
  -m "${BASE_MODEL}" -algo FedAvg \
  -ncl 10 -nc "${NUM_CLIENTS}" -jr "${JOIN_RATIO}" -lbs "${BATCH_SIZE}" \
  -lr "${LOCAL_LR}" -ls "${LOCAL_EPOCHS}" \
  -gr "${GLOBAL_ROUNDS}" -eg "${EVAL_GAP}" \
  -go main_blend_fedavg \
  --num_adv_clients "${ADV_CLIENTS}" \
  > "${TRAIN_LOG_DIR}/main_blend_fedavg.log" 2>&1

"${PY}" -u main.py \
  -dev cuda -did 1 \
  -data "${BLEND_DATASET}" \
  -m "${BASE_MODEL}" -algo FedRep \
  -ncl 10 -nc "${NUM_CLIENTS}" -jr "${JOIN_RATIO}" -lbs "${BATCH_SIZE}" \
  -lr "${LOCAL_LR}" -lr_head "${LR_HEAD}" -ls "${LOCAL_EPOCHS}" -pls "${PLOCAL_EPOCHS}" \
  -gr "${GLOBAL_ROUNDS}" -eg "${EVAL_GAP}" \
  -go main_blend_fedrep \
  --num_adv_clients "${ADV_CLIENTS}" \
  > "${TRAIN_LOG_DIR}/main_blend_fedrep.log" 2>&1

"${PY}" -u main.py \
  -dev cuda -did 1 \
  -data "${BLEND_DATASET}" \
  -m "${BASE_MODEL}" -algo FedTrimmed \
  -ncl 10 -nc "${NUM_CLIENTS}" -jr "${JOIN_RATIO}" -lbs "${BATCH_SIZE}" \
  -lr "${LOCAL_LR}" -ls "${LOCAL_EPOCHS}" \
  -gr "${GLOBAL_ROUNDS}" -eg "${EVAL_GAP}" \
  -go main_blend_fedtrimmed \
  --num_adv_clients "${ADV_CLIENTS}" \
  > "${TRAIN_LOG_DIR}/main_blend_fedtrimmed.log" 2>&1
EOF

  cat > "${QUEUE_DIR}/gpu2_queue.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${SRC}"

"${PY}" -u main.py \
  -dev cuda -did 2 \
  -data "${SIG_DATASET}" \
  -m "${BASE_MODEL}" -algo FedAvg \
  -ncl 10 -nc "${NUM_CLIENTS}" -jr "${JOIN_RATIO}" -lbs "${BATCH_SIZE}" \
  -lr "${LOCAL_LR}" -ls "${LOCAL_EPOCHS}" \
  -gr "${GLOBAL_ROUNDS}" -eg "${EVAL_GAP}" \
  -go main_sig_fedavg \
  --num_adv_clients "${ADV_CLIENTS}" \
  > "${TRAIN_LOG_DIR}/main_sig_fedavg.log" 2>&1

"${PY}" -u main.py \
  -dev cuda -did 2 \
  -data "${SIG_DATASET}" \
  -m "${BASE_MODEL}" -algo FedRep \
  -ncl 10 -nc "${NUM_CLIENTS}" -jr "${JOIN_RATIO}" -lbs "${BATCH_SIZE}" \
  -lr "${LOCAL_LR}" -lr_head "${LR_HEAD}" -ls "${LOCAL_EPOCHS}" -pls "${PLOCAL_EPOCHS}" \
  -gr "${GLOBAL_ROUNDS}" -eg "${EVAL_GAP}" \
  -go main_sig_fedrep \
  --num_adv_clients "${ADV_CLIENTS}" \
  > "${TRAIN_LOG_DIR}/main_sig_fedrep.log" 2>&1

"${PY}" -u main.py \
  -dev cuda -did 2 \
  -data "${SIG_DATASET}" \
  -m "${BASE_MODEL}" -algo FedTrimmed \
  -ncl 10 -nc "${NUM_CLIENTS}" -jr "${JOIN_RATIO}" -lbs "${BATCH_SIZE}" \
  -lr "${LOCAL_LR}" -ls "${LOCAL_EPOCHS}" \
  -gr "${GLOBAL_ROUNDS}" -eg "${EVAL_GAP}" \
  -go main_sig_fedtrimmed \
  --num_adv_clients "${ADV_CLIENTS}" \
  > "${TRAIN_LOG_DIR}/main_sig_fedtrimmed.log" 2>&1
EOF

  cat > "${QUEUE_DIR}/gpu3_queue.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${SRC}"

"${PY}" -u main.py \
  -dev cuda -did 3 \
  -data "${BADNET_DATASET}" \
  -m "${BASE_MODEL}" -algo FedCLCM \
  -ncl 10 -nc "${NUM_CLIENTS}" -jr "${JOIN_RATIO}" -lbs "${BATCH_SIZE}" \
  -lr "${LOCAL_LR}" -lr_head "${LR_HEAD}" -ls "${LOCAL_EPOCHS}" -pls "${PLOCAL_EPOCHS}" \
  -gr "${GLOBAL_ROUNDS}" -eg "${EVAL_GAP}" \
  -go main_badnet_fedclcm \
  --num_adv_clients "${ADV_CLIENTS}" \
  --rt_beta 0.2 --lambda_cl 0.20 --aug_strength 0.1 \
  --adv_eps 0 --adv_num_iter 0 --mask_tau 12.0 --mask_alpha 0.70 \
  > "${TRAIN_LOG_DIR}/main_badnet_fedclcm.log" 2>&1

"${PY}" -u main.py \
  -dev cuda -did 3 \
  -data "${BLEND_DATASET}" \
  -m "${BASE_MODEL}" -algo FedCLCM \
  -ncl 10 -nc "${NUM_CLIENTS}" -jr "${JOIN_RATIO}" -lbs "${BATCH_SIZE}" \
  -lr "${LOCAL_LR}" -lr_head "${LR_HEAD}" -ls "${LOCAL_EPOCHS}" -pls "${PLOCAL_EPOCHS}" \
  -gr "${GLOBAL_ROUNDS}" -eg "${EVAL_GAP}" \
  -go main_blend_fedclcm \
  --num_adv_clients "${ADV_CLIENTS}" \
  --rt_beta 0.2 --lambda_cl 0.20 --aug_strength 0.1 \
  --adv_eps 0 --adv_num_iter 0 --mask_tau 12.0 --mask_alpha 0.70 \
  > "${TRAIN_LOG_DIR}/main_blend_fedclcm.log" 2>&1

"${PY}" -u main.py \
  -dev cuda -did 3 \
  -data "${SIG_DATASET}" \
  -m "${BASE_MODEL}" -algo FedCLCM \
  -ncl 10 -nc "${NUM_CLIENTS}" -jr "${JOIN_RATIO}" -lbs "${BATCH_SIZE}" \
  -lr "${LOCAL_LR}" -lr_head "${LR_HEAD}" -ls "${LOCAL_EPOCHS}" -pls "${PLOCAL_EPOCHS}" \
  -gr "${GLOBAL_ROUNDS}" -eg "${EVAL_GAP}" \
  -go main_sig_fedclcm \
  --num_adv_clients "${ADV_CLIENTS}" \
  --rt_beta 0.2 --lambda_cl 0.20 --aug_strength 0.1 \
  --adv_eps 0 --adv_num_iter 0 --mask_tau 12.0 --mask_alpha 0.70 \
  > "${TRAIN_LOG_DIR}/main_sig_fedclcm.log" 2>&1
EOF

  chmod +x "${QUEUE_DIR}/gpu0_queue.sh" "${QUEUE_DIR}/gpu1_queue.sh" "${QUEUE_DIR}/gpu2_queue.sh" "${QUEUE_DIR}/gpu3_queue.sh"
}

main() {
  ensure_mode
  ensure_gpus

  gen_dataset_if_missing generate_Cifar10_badnet.py "${BADNET_DATASET}" generate_badnet.log
  gen_dataset_if_missing generate_Cifar10_blend.py "${BLEND_DATASET}" generate_blend.log \
    --blend-alpha "${BLEND_ALPHA}"
  gen_dataset_if_missing generate_Cifar10_sig.py "${SIG_DATASET}" generate_sig.log \
    --sig-delta "${SIG_DELTA}" \
    --sig-f "${SIG_F}" \
    --sig-label-mode dirty

  write_readme

  if [[ "${MODE}" == "gate" ]]; then
    write_manifest_gate
    write_gate_queues
    launch_queue gpu0_queue
    launch_queue gpu1_queue
    launch_queue gpu2_queue
  else
    write_manifest_main
    write_main_queues
    launch_queue gpu0_queue
    launch_queue gpu1_queue
    launch_queue gpu2_queue
    launch_queue gpu3_queue
  fi

  echo "[READY] ${MODE} launched under ${RUN_ROOT}"
  echo "[CHECK] logs: ${TRAIN_LOG_DIR}"
}

main "$@"
