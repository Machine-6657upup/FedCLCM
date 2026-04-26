#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC="${ROOT}/src"
RUNS_DIR="${ROOT}/runs"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUNS_DIR}/${RUN_TS}_stage3_static_paper_common"
DATASET_LOG_DIR="${RUN_ROOT}/dataset_logs"
QUEUE_DIR="${RUN_ROOT}/queue_scripts"
LAUNCHER_LOG_DIR="${RUN_ROOT}/launcher_logs"
TRAIN_LOG_DIR="${RUN_ROOT}/train_logs"
PID_DIR="${RUN_ROOT}/pids"
PY="${PYTHON_BIN:-/home/huangtu/miniconda3/envs/torch/bin/python}"
WRAPPER="${ROOT}/scripts/generate_dataset_via_existing.py"
GEN_DIR="${SRC}/dataset/utils"
RAWDATA="${SRC}/dataset/rawdata"

NUM_CLIENTS="${NUM_CLIENTS:-100}"
JOIN_RATIO="${JOIN_RATIO:-0.1}"
LOCAL_LR="${LOCAL_LR:-0.1}"
LR_HEAD="${LR_HEAD:-0.1}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
PLOCAL_EPOCHS="${PLOCAL_EPOCHS:-1}"
ADV_CLIENTS="${ADV_CLIENTS:-10}"
BACKDOOR_RATE="${BACKDOOR_RATE:-0.2}"
GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-600}"
BASE_MODEL="${BASE_MODEL:-ResNet18}"
ENABLE_FEDCLCM="${ENABLE_FEDCLCM:-1}"
SIG_DELTA="${SIG_DELTA:-0.11764705882352941}"
SIG_F="${SIG_F:-6}"
BLEND_ALPHA="${BLEND_ALPHA:-0.2}"

BADNET_DATASET="Cifar10_dir0.5_bdoor${BACKDOOR_RATE}_nclient_${NUM_CLIENTS}_badnet_adv${ADV_CLIENTS}"
BLEND_DATASET="Cifar10_dir0.5_bdoor${BACKDOOR_RATE}_nclient_${NUM_CLIENTS}_blend_adv${ADV_CLIENTS}"
SIG_DATASET="Cifar10_dir0.5_bdoor${BACKDOOR_RATE}_nclient_${NUM_CLIENTS}_sig_adv${ADV_CLIENTS}"

mkdir -p "${DATASET_LOG_DIR}" "${QUEUE_DIR}" "${LAUNCHER_LOG_DIR}" "${TRAIN_LOG_DIR}" "${PID_DIR}"

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

cat > "${RUN_ROOT}/README.txt" <<EOF
Stage 3: static attacks under paper-common regime
Timestamp: ${RUN_TS}

Purpose:
- align static experiments with a paper-common PFL regime
- validate stronger dirty-label Blend and SIG implementations
- compare FedAvg and FedRep first, then FedCLCM if a GPU is available

Common regime:
- num_clients = ${NUM_CLIENTS}
- join_ratio = ${JOIN_RATIO}
- lr = ${LOCAL_LR}
- lr_head = ${LR_HEAD}
- local_epochs = ${LOCAL_EPOCHS}
- plocal_epochs = ${PLOCAL_EPOCHS}
- model = ${BASE_MODEL}
- num_adv_clients = ${ADV_CLIENTS}
- global_rounds = ${GLOBAL_ROUNDS}

Attack-specific generation defaults:
- blend_alpha = ${BLEND_ALPHA}
- sig_delta = ${SIG_DELTA}
- sig_f = ${SIG_F}
- sig_label_mode = dirty
EOF

cat > "${RUN_ROOT}/manifest.tsv" <<EOF
task_tag	gpu	algorithm	dataset	model	join_ratio	lr	lr_head	rounds
s3_avg_badnet	0	FedAvg	${BADNET_DATASET}	${BASE_MODEL}	${JOIN_RATIO}	${LOCAL_LR}	-	${GLOBAL_ROUNDS}
s3_rep_badnet	0	FedRep	${BADNET_DATASET}	${BASE_MODEL}	${JOIN_RATIO}	${LOCAL_LR}	${LR_HEAD}	${GLOBAL_ROUNDS}
s3_avg_blend	1	FedAvg	${BLEND_DATASET}	${BASE_MODEL}	${JOIN_RATIO}	${LOCAL_LR}	-	${GLOBAL_ROUNDS}
s3_rep_blend	1	FedRep	${BLEND_DATASET}	${BASE_MODEL}	${JOIN_RATIO}	${LOCAL_LR}	${LR_HEAD}	${GLOBAL_ROUNDS}
s3_avg_sig	2	FedAvg	${SIG_DATASET}	${BASE_MODEL}	${JOIN_RATIO}	${LOCAL_LR}	-	${GLOBAL_ROUNDS}
s3_rep_sig	2	FedRep	${SIG_DATASET}	${BASE_MODEL}	${JOIN_RATIO}	${LOCAL_LR}	${LR_HEAD}	${GLOBAL_ROUNDS}
s3_clcm_badnet	3	FedCLCM	${BADNET_DATASET}	${BASE_MODEL}	${JOIN_RATIO}	${LOCAL_LR}	-	${GLOBAL_ROUNDS}
s3_clcm_blend	3	FedCLCM	${BLEND_DATASET}	${BASE_MODEL}	${JOIN_RATIO}	${LOCAL_LR}	-	${GLOBAL_ROUNDS}
s3_clcm_sig	3	FedCLCM	${SIG_DATASET}	${BASE_MODEL}	${JOIN_RATIO}	${LOCAL_LR}	-	${GLOBAL_ROUNDS}
EOF

gen_dataset_if_missing generate_Cifar10_badnet.py "${BADNET_DATASET}" generate_badnet.log
gen_dataset_if_missing generate_Cifar10_blend.py "${BLEND_DATASET}" generate_blend.log \
  --blend-alpha "${BLEND_ALPHA}"
gen_dataset_if_missing generate_Cifar10_sig.py "${SIG_DATASET}" generate_sig.log \
  --sig-delta "${SIG_DELTA}" \
  --sig-f "${SIG_F}" \
  --sig-label-mode dirty

cat > "${QUEUE_DIR}/gpu0_queue.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${SRC}"
PY="${PY}"

"\${PY}" -u main.py \
  -dev cuda -did 0 \
  -data ${BADNET_DATASET} \
  -m ${BASE_MODEL} -algo FedAvg \
  -ncl 10 -nc ${NUM_CLIENTS} -jr ${JOIN_RATIO} -lbs 64 \
  -lr ${LOCAL_LR} -ls ${LOCAL_EPOCHS} \
  -gr ${GLOBAL_ROUNDS} -eg 1 \
  -go s3_avg_badnet \
  --num_adv_clients ${ADV_CLIENTS} \
  > "${TRAIN_LOG_DIR}/s3_avg_badnet.log" 2>&1

"\${PY}" -u main.py \
  -dev cuda -did 0 \
  -data ${BADNET_DATASET} \
  -m ${BASE_MODEL} -algo FedRep \
  -ncl 10 -nc ${NUM_CLIENTS} -jr ${JOIN_RATIO} -lbs 64 \
  -lr ${LOCAL_LR} -lr_head ${LR_HEAD} -ls ${LOCAL_EPOCHS} -pls ${PLOCAL_EPOCHS} \
  -gr ${GLOBAL_ROUNDS} -eg 1 \
  -go s3_rep_badnet \
  --num_adv_clients ${ADV_CLIENTS} \
  > "${TRAIN_LOG_DIR}/s3_rep_badnet.log" 2>&1
EOF

cat > "${QUEUE_DIR}/gpu1_queue.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${SRC}"
PY="${PY}"

"\${PY}" -u main.py \
  -dev cuda -did 1 \
  -data ${BLEND_DATASET} \
  -m ${BASE_MODEL} -algo FedAvg \
  -ncl 10 -nc ${NUM_CLIENTS} -jr ${JOIN_RATIO} -lbs 64 \
  -lr ${LOCAL_LR} -ls ${LOCAL_EPOCHS} \
  -gr ${GLOBAL_ROUNDS} -eg 1 \
  -go s3_avg_blend \
  --num_adv_clients ${ADV_CLIENTS} \
  > "${TRAIN_LOG_DIR}/s3_avg_blend.log" 2>&1

"\${PY}" -u main.py \
  -dev cuda -did 1 \
  -data ${BLEND_DATASET} \
  -m ${BASE_MODEL} -algo FedRep \
  -ncl 10 -nc ${NUM_CLIENTS} -jr ${JOIN_RATIO} -lbs 64 \
  -lr ${LOCAL_LR} -lr_head ${LR_HEAD} -ls ${LOCAL_EPOCHS} -pls ${PLOCAL_EPOCHS} \
  -gr ${GLOBAL_ROUNDS} -eg 1 \
  -go s3_rep_blend \
  --num_adv_clients ${ADV_CLIENTS} \
  > "${TRAIN_LOG_DIR}/s3_rep_blend.log" 2>&1
EOF

cat > "${QUEUE_DIR}/gpu2_queue.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${SRC}"
PY="${PY}"

"\${PY}" -u main.py \
  -dev cuda -did 2 \
  -data ${SIG_DATASET} \
  -m ${BASE_MODEL} -algo FedAvg \
  -ncl 10 -nc ${NUM_CLIENTS} -jr ${JOIN_RATIO} -lbs 64 \
  -lr ${LOCAL_LR} -ls ${LOCAL_EPOCHS} \
  -gr ${GLOBAL_ROUNDS} -eg 1 \
  -go s3_avg_sig \
  --num_adv_clients ${ADV_CLIENTS} \
  > "${TRAIN_LOG_DIR}/s3_avg_sig.log" 2>&1

"\${PY}" -u main.py \
  -dev cuda -did 2 \
  -data ${SIG_DATASET} \
  -m ${BASE_MODEL} -algo FedRep \
  -ncl 10 -nc ${NUM_CLIENTS} -jr ${JOIN_RATIO} -lbs 64 \
  -lr ${LOCAL_LR} -lr_head ${LR_HEAD} -ls ${LOCAL_EPOCHS} -pls ${PLOCAL_EPOCHS} \
  -gr ${GLOBAL_ROUNDS} -eg 1 \
  -go s3_rep_sig \
  --num_adv_clients ${ADV_CLIENTS} \
  > "${TRAIN_LOG_DIR}/s3_rep_sig.log" 2>&1
EOF

if [[ "${ENABLE_FEDCLCM}" == "1" ]]; then
  cat > "${QUEUE_DIR}/gpu3_queue.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${SRC}"
PY="${PY}"

"\${PY}" -u main.py \
  -dev cuda -did 3 \
  -data ${BADNET_DATASET} \
  -m ${BASE_MODEL} -algo FedCLCM \
  -ncl 10 -nc ${NUM_CLIENTS} -jr ${JOIN_RATIO} -lbs 64 \
  -lr ${LOCAL_LR} -ls ${LOCAL_EPOCHS} \
  -gr ${GLOBAL_ROUNDS} -eg 1 \
  -go s3_clcm_badnet \
  --num_adv_clients ${ADV_CLIENTS} \
  > "${TRAIN_LOG_DIR}/s3_clcm_badnet.log" 2>&1

"\${PY}" -u main.py \
  -dev cuda -did 3 \
  -data ${BLEND_DATASET} \
  -m ${BASE_MODEL} -algo FedCLCM \
  -ncl 10 -nc ${NUM_CLIENTS} -jr ${JOIN_RATIO} -lbs 64 \
  -lr ${LOCAL_LR} -ls ${LOCAL_EPOCHS} \
  -gr ${GLOBAL_ROUNDS} -eg 1 \
  -go s3_clcm_blend \
  --num_adv_clients ${ADV_CLIENTS} \
  > "${TRAIN_LOG_DIR}/s3_clcm_blend.log" 2>&1

"\${PY}" -u main.py \
  -dev cuda -did 3 \
  -data ${SIG_DATASET} \
  -m ${BASE_MODEL} -algo FedCLCM \
  -ncl 10 -nc ${NUM_CLIENTS} -jr ${JOIN_RATIO} -lbs 64 \
  -lr ${LOCAL_LR} -ls ${LOCAL_EPOCHS} \
  -gr ${GLOBAL_ROUNDS} -eg 1 \
  -go s3_clcm_sig \
  --num_adv_clients ${ADV_CLIENTS} \
  > "${TRAIN_LOG_DIR}/s3_clcm_sig.log" 2>&1
EOF
  chmod +x "${QUEUE_DIR}/gpu3_queue.sh"
  launch_queue gpu3_queue
fi

chmod +x "${QUEUE_DIR}/gpu0_queue.sh" \
         "${QUEUE_DIR}/gpu1_queue.sh" \
         "${QUEUE_DIR}/gpu2_queue.sh"

launch_queue gpu0_queue
launch_queue gpu1_queue
launch_queue gpu2_queue

echo "[READY] stage3 launched under ${RUN_ROOT}"
