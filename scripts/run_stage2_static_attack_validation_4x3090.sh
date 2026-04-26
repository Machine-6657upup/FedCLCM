#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC="${ROOT}/src"
RUNS_DIR="${ROOT}/runs"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUNS_DIR}/${RUN_TS}_stage2_static_attack_validation"
DATASET_LOG_DIR="${RUN_ROOT}/dataset_logs"
QUEUE_DIR="${RUN_ROOT}/queue_scripts"
LAUNCHER_LOG_DIR="${RUN_ROOT}/launcher_logs"
TRAIN_LOG_DIR="${RUN_ROOT}/train_logs"
PID_DIR="${RUN_ROOT}/pids"
PY="${PYTHON_BIN:-/home/huangtu/miniconda3/envs/torch/bin/python}"
WRAPPER="${ROOT}/scripts/generate_dataset_via_existing.py"
GEN_DIR="${SRC}/dataset/utils"
RAWDATA="${SRC}/dataset/rawdata"
GLOBAL_ROUNDS_BADNET="${GLOBAL_ROUNDS_BADNET:-800}"
GLOBAL_ROUNDS_WEAK="${GLOBAL_ROUNDS_WEAK:-600}"

# stage1 winner defaults; override after stage1 if needed
BASE_MODEL="${BASE_MODEL:-ResNet18}"
BASE_JOIN_RATIO="${BASE_JOIN_RATIO:-1.0}"
BASE_LR="${BASE_LR:-0.1}"
BASE_LR_HEAD="${BASE_LR_HEAD:-0.1}"

mkdir -p "${DATASET_LOG_DIR}" "${QUEUE_DIR}" "${LAUNCHER_LOG_DIR}" "${TRAIN_LOG_DIR}" "${PID_DIR}"

gen_dataset_if_missing() {
  local generator="$1"
  local dataset_name="$2"
  local backdoor_rate="$3"
  local adv_num="$4"
  local log_name="$5"
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
    --num-clients 40 \
    --backdoor-rate "${backdoor_rate}" \
    --adversary-num "${adv_num}" \
    --target-y 0 \
    --alpha 0.5 \
    --train-ratio 0.8 \
    --batch-size 10 \
    --partition dir \
    --niid \
    --balance \
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
Stage 2: static attack validation on locked baseline settings
Timestamp: ${RUN_TS}

Purpose:
- verify that BadNet, Blend, and SIG are not trivially weak
- compare FedAvg and FedRep on the same locked baseline settings
- avoid interpreting low ASR as strong defense before attack strength is validated

Baseline settings:
- model = ${BASE_MODEL}
- join_ratio = ${BASE_JOIN_RATIO}
- lr = ${BASE_LR}
- lr_head = ${BASE_LR_HEAD}
- num_clients = 40
- num_adv_clients = 5
EOF

cat > "${RUN_ROOT}/manifest.tsv" <<EOF
task_tag	gpu	algorithm	dataset	model	join_ratio	lr	lr_head	rounds
s2_avg_badnet	0	FedAvg	Cifar10_dir0.5_bdoor0.2_nclient_40_badnet_adv5	${BASE_MODEL}	${BASE_JOIN_RATIO}	${BASE_LR}	-	${GLOBAL_ROUNDS_BADNET}
s2_rep_badnet	1	FedRep	Cifar10_dir0.5_bdoor0.2_nclient_40_badnet_adv5	${BASE_MODEL}	${BASE_JOIN_RATIO}	${BASE_LR}	${BASE_LR_HEAD}	${GLOBAL_ROUNDS_BADNET}
s2_avg_blend	2	FedAvg	Cifar10_dir0.5_bdoor0.2_nclient_40_blend_adv5	${BASE_MODEL}	${BASE_JOIN_RATIO}	${BASE_LR}	-	${GLOBAL_ROUNDS_WEAK}
s2_rep_blend	3	FedRep	Cifar10_dir0.5_bdoor0.2_nclient_40_blend_adv5	${BASE_MODEL}	${BASE_JOIN_RATIO}	${BASE_LR}	${BASE_LR_HEAD}	${GLOBAL_ROUNDS_WEAK}
s2_avg_sig	0	FedAvg	Cifar10_dir0.5_bdoor0.2_nclient_40_sig_adv5	${BASE_MODEL}	${BASE_JOIN_RATIO}	${BASE_LR}	-	${GLOBAL_ROUNDS_WEAK}
s2_rep_sig	1	FedRep	Cifar10_dir0.5_bdoor0.2_nclient_40_sig_adv5	${BASE_MODEL}	${BASE_JOIN_RATIO}	${BASE_LR}	${BASE_LR_HEAD}	${GLOBAL_ROUNDS_WEAK}
EOF

gen_dataset_if_missing generate_Cifar10_badnet.py Cifar10_dir0.5_bdoor0.2_nclient_40_badnet_adv5 0.2 5 generate_badnet.log
gen_dataset_if_missing generate_Cifar10_blend.py Cifar10_dir0.5_bdoor0.2_nclient_40_blend_adv5 0.2 5 generate_blend.log
gen_dataset_if_missing generate_Cifar10_sig.py Cifar10_dir0.5_bdoor0.2_nclient_40_sig_adv5 0.2 5 generate_sig.log

cat > "${QUEUE_DIR}/gpu0_queue.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${SRC}"
PY="${PY}"

"\${PY}" -u main.py \
  -dev cuda -did 0 \
  -data Cifar10_dir0.5_bdoor0.2_nclient_40_badnet_adv5 \
  -m ${BASE_MODEL} -algo FedAvg \
  -ncl 10 -nc 40 -jr ${BASE_JOIN_RATIO} -lbs 64 \
  -lr ${BASE_LR} -ls 1 \
  -gr ${GLOBAL_ROUNDS_BADNET} -eg 1 \
  -go s2_avg_badnet \
  --num_adv_clients 5 \
  > "${TRAIN_LOG_DIR}/s2_avg_badnet.log" 2>&1

"\${PY}" -u main.py \
  -dev cuda -did 0 \
  -data Cifar10_dir0.5_bdoor0.2_nclient_40_sig_adv5 \
  -m ${BASE_MODEL} -algo FedAvg \
  -ncl 10 -nc 40 -jr ${BASE_JOIN_RATIO} -lbs 64 \
  -lr ${BASE_LR} -ls 1 \
  -gr ${GLOBAL_ROUNDS_WEAK} -eg 1 \
  -go s2_avg_sig \
  --num_adv_clients 5 \
  > "${TRAIN_LOG_DIR}/s2_avg_sig.log" 2>&1
EOF

cat > "${QUEUE_DIR}/gpu1_queue.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${SRC}"
PY="${PY}"

"\${PY}" -u main.py \
  -dev cuda -did 1 \
  -data Cifar10_dir0.5_bdoor0.2_nclient_40_badnet_adv5 \
  -m ${BASE_MODEL} -algo FedRep \
  -ncl 10 -nc 40 -jr ${BASE_JOIN_RATIO} -lbs 64 \
  -lr ${BASE_LR} -lr_head ${BASE_LR_HEAD} -ls 1 -pls 1 \
  -gr ${GLOBAL_ROUNDS_BADNET} -eg 1 \
  -go s2_rep_badnet \
  --num_adv_clients 5 \
  > "${TRAIN_LOG_DIR}/s2_rep_badnet.log" 2>&1

"\${PY}" -u main.py \
  -dev cuda -did 1 \
  -data Cifar10_dir0.5_bdoor0.2_nclient_40_sig_adv5 \
  -m ${BASE_MODEL} -algo FedRep \
  -ncl 10 -nc 40 -jr ${BASE_JOIN_RATIO} -lbs 64 \
  -lr ${BASE_LR} -lr_head ${BASE_LR_HEAD} -ls 1 -pls 1 \
  -gr ${GLOBAL_ROUNDS_WEAK} -eg 1 \
  -go s2_rep_sig \
  --num_adv_clients 5 \
  > "${TRAIN_LOG_DIR}/s2_rep_sig.log" 2>&1
EOF

cat > "${QUEUE_DIR}/gpu2_queue.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${SRC}"
PY="${PY}"

"\${PY}" -u main.py \
  -dev cuda -did 2 \
  -data Cifar10_dir0.5_bdoor0.2_nclient_40_blend_adv5 \
  -m ${BASE_MODEL} -algo FedAvg \
  -ncl 10 -nc 40 -jr ${BASE_JOIN_RATIO} -lbs 64 \
  -lr ${BASE_LR} -ls 1 \
  -gr ${GLOBAL_ROUNDS_WEAK} -eg 1 \
  -go s2_avg_blend \
  --num_adv_clients 5 \
  > "${TRAIN_LOG_DIR}/s2_avg_blend.log" 2>&1
EOF

cat > "${QUEUE_DIR}/gpu3_queue.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${SRC}"
PY="${PY}"

"\${PY}" -u main.py \
  -dev cuda -did 3 \
  -data Cifar10_dir0.5_bdoor0.2_nclient_40_blend_adv5 \
  -m ${BASE_MODEL} -algo FedRep \
  -ncl 10 -nc 40 -jr ${BASE_JOIN_RATIO} -lbs 64 \
  -lr ${BASE_LR} -lr_head ${BASE_LR_HEAD} -ls 1 -pls 1 \
  -gr ${GLOBAL_ROUNDS_WEAK} -eg 1 \
  -go s2_rep_blend \
  --num_adv_clients 5 \
  > "${TRAIN_LOG_DIR}/s2_rep_blend.log" 2>&1
EOF

chmod +x "${QUEUE_DIR}/gpu0_queue.sh" \
         "${QUEUE_DIR}/gpu1_queue.sh" \
         "${QUEUE_DIR}/gpu2_queue.sh" \
         "${QUEUE_DIR}/gpu3_queue.sh"

launch_queue gpu0_queue
launch_queue gpu1_queue
launch_queue gpu2_queue
launch_queue gpu3_queue

echo "[READY] stage2 launched under ${RUN_ROOT}"
