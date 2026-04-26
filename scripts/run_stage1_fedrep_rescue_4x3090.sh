#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC="${ROOT}/src"
RUNS_DIR="${ROOT}/runs"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUNS_DIR}/${RUN_TS}_stage1_fedrep_rescue"
DATASET_LOG_DIR="${RUN_ROOT}/dataset_logs"
QUEUE_DIR="${RUN_ROOT}/queue_scripts"
LAUNCHER_LOG_DIR="${RUN_ROOT}/launcher_logs"
TRAIN_LOG_DIR="${RUN_ROOT}/train_logs"
PID_DIR="${RUN_ROOT}/pids"
PY="${PYTHON_BIN:-/home/huangtu/miniconda3/envs/torch/bin/python}"
WRAPPER="${ROOT}/scripts/generate_dataset_via_existing.py"
GEN_DIR="${SRC}/dataset/utils"
RAWDATA="${SRC}/dataset/rawdata"
DATASET_NAME="${DATASET_NAME:-Cifar10_dir0.5_bdoor0.2_nclient_40_badnet_adv5}"
GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-800}"

mkdir -p "${DATASET_LOG_DIR}" "${QUEUE_DIR}" "${LAUNCHER_LOG_DIR}" "${TRAIN_LOG_DIR}" "${PID_DIR}"

gen_dataset_if_missing() {
  local config_path="${SRC}/dataset/${DATASET_NAME}/config.json"
  if [[ -f "${config_path}" ]]; then
    echo "[DATASET] exists ${DATASET_NAME}"
    return 0
  fi

  echo "[DATASET] generating ${DATASET_NAME}"
  "${PY}" "${WRAPPER}" \
    --generator "${GEN_DIR}/generate_Cifar10_badnet.py" \
    --dir-path "${SRC}/dataset/${DATASET_NAME}" \
    --rawdata-path "${RAWDATA}" \
    --num-clients 40 \
    --backdoor-rate 0.2 \
    --adversary-num 5 \
    --target-y 0 \
    --alpha 0.5 \
    --train-ratio 0.8 \
    --batch-size 10 \
    --partition dir \
    --niid \
    --balance \
    > "${DATASET_LOG_DIR}/generate_badnet_40c_adv5.log" 2>&1
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
Stage 1: FedRep baseline rescue
Timestamp: ${RUN_TS}

Purpose:
- recover a strong and reproducible FedRep baseline
- focus on the historical 40-client badnet line
- avoid mixing this stage with the current 100-client thesis mainline

Dataset:
- ${DATASET_NAME}

Fixed settings:
- num_clients = 40
- num_adv_clients = 5
- local_epochs = 1
- plocal_epochs = 1
- batch_size = 64
- global_rounds = ${GLOBAL_ROUNDS}

Sweep factors:
- model: ResNet18 vs ResNetP
- join_ratio: 1.0 vs 0.1
- learning-rate family: narrow manual rescue sweep
EOF

cat > "${RUN_ROOT}/manifest.tsv" <<'EOF'
task_tag	gpu	model	join_ratio	lr	lr_head
s1_r18_j1_lr0p1_h0p1	0	ResNet18	1.0	0.1	0.1
s1_r18_j1_lr0p03_h0p03	1	ResNet18	1.0	0.03	0.03
s1_r18_j0p1_lr0p1_h0p1	2	ResNet18	0.1	0.1	0.1
s1_r18_j1_lr0p01_h0p01	3	ResNet18	1.0	0.01	0.01
s1_rp_j1_lr0p003_h0p01	0	ResNetP	1.0	0.003	0.01
s1_rp_j1_lr0p01_h0p03	1	ResNetP	1.0	0.01	0.03
s1_rp_j0p1_lr0p003_h0p01	2	ResNetP	0.1	0.003	0.01
s1_rp_j0p1_lr0p01_h0p03	3	ResNetP	0.1	0.01	0.03
EOF

gen_dataset_if_missing

make_queue() {
  local queue_name="$1"
  local gpu="$2"
  shift 2

  cat > "${QUEUE_DIR}/${queue_name}.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${SRC}"
PY="${PY}"
EOF

  while [[ "$#" -gt 0 ]]; do
    local task_tag="$1"
    local model="$2"
    local jr="$3"
    local lr="$4"
    local lr_head="$5"
    shift 5

    cat >> "${QUEUE_DIR}/${queue_name}.sh" <<EOF

"\${PY}" -u main.py \
  -dev cuda -did ${gpu} \
  -data ${DATASET_NAME} \
  -m ${model} -algo FedRep \
  -ncl 10 -nc 40 -jr ${jr} -lbs 64 \
  -lr ${lr} -lr_head ${lr_head} -ls 1 -pls 1 \
  -gr ${GLOBAL_ROUNDS} -eg 1 \
  -go ${task_tag} \
  --num_adv_clients 5 \
  > "${TRAIN_LOG_DIR}/${task_tag}.log" 2>&1
EOF
  done

  chmod +x "${QUEUE_DIR}/${queue_name}.sh"
}

make_queue gpu0_queue 0 \
  s1_r18_j1_lr0p1_h0p1 ResNet18 1.0 0.1 0.1 \
  s1_rp_j1_lr0p003_h0p01 ResNetP 1.0 0.003 0.01

make_queue gpu1_queue 1 \
  s1_r18_j1_lr0p03_h0p03 ResNet18 1.0 0.03 0.03 \
  s1_rp_j1_lr0p01_h0p03 ResNetP 1.0 0.01 0.03

make_queue gpu2_queue 2 \
  s1_r18_j0p1_lr0p1_h0p1 ResNet18 0.1 0.1 0.1 \
  s1_rp_j0p1_lr0p003_h0p01 ResNetP 0.1 0.003 0.01

make_queue gpu3_queue 3 \
  s1_r18_j1_lr0p01_h0p01 ResNet18 1.0 0.01 0.01 \
  s1_rp_j0p1_lr0p01_h0p03 ResNetP 0.1 0.01 0.03

launch_queue gpu0_queue
launch_queue gpu1_queue
launch_queue gpu2_queue
launch_queue gpu3_queue

echo "[READY] stage1 launched under ${RUN_ROOT}"
