#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC="${ROOT}/src"
RUN_ROOT="${RUN_ROOT:-${ROOT}/thesis_formal_logs/final_static_main_20260424}"
STATIC_DIR="${RUN_ROOT}/static"
QUEUE_DIR="${RUN_ROOT}/queue_scripts_trimmed"
LAUNCHER_DIR="${RUN_ROOT}/launcher_logs_trimmed"
PID_DIR="${RUN_ROOT}/pids_trimmed"
DATASET_LOG_DIR="${RUN_ROOT}/dataset_logs_trimmed"

PY="${PYTHON_BIN:-/home/huangtu/miniconda3/envs/torch/bin/python}"
WRAPPER="${ROOT}/scripts/generate_dataset_via_existing.py"
GEN_DIR="${SRC}/dataset/utils"
RAWDATA="${SRC}/dataset/rawdata"

GPU_BADNET="${GPU_BADNET:-0}"
GPU_BLEND="${GPU_BLEND:-1}"
GPU_SIG="${GPU_SIG:-2}"

NUM_CLIENTS="${NUM_CLIENTS:-100}"
JOIN_RATIO="${JOIN_RATIO:-0.1}"
LOCAL_LR="${LOCAL_LR:-0.1}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-600}"
EVAL_GAP="${EVAL_GAP:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BASE_MODEL="${BASE_MODEL:-ResNet18}"
ADV_CLIENTS="${ADV_CLIENTS:-10}"
BACKDOOR_RATE="${BACKDOOR_RATE:-0.2}"
BLEND_ALPHA="${BLEND_ALPHA:-0.2}"
SIG_DELTA="${SIG_DELTA:-0.11764705882352941}"
SIG_F="${SIG_F:-6}"

BADNET_DATASET="Cifar10_dir0.5_bdoor${BACKDOOR_RATE}_nclient_${NUM_CLIENTS}_badnet_adv${ADV_CLIENTS}"
BLEND_DATASET="Cifar10_dir0.5_bdoor${BACKDOOR_RATE}_nclient_${NUM_CLIENTS}_blend_adv${ADV_CLIENTS}"
SIG_DATASET="Cifar10_dir0.5_bdoor${BACKDOOR_RATE}_nclient_${NUM_CLIENTS}_sig_adv${ADV_CLIENTS}"

mkdir -p "${RUN_ROOT}" "${STATIC_DIR}" "${QUEUE_DIR}" "${LAUNCHER_DIR}" "${PID_DIR}" "${DATASET_LOG_DIR}"

ensure_gpus() {
  local count
  count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
  if [[ -z "${count}" || "${count}" -lt 3 ]]; then
    echo "[FATAL] need at least 3 visible GPUs for the trimmed-only launcher." >&2
    exit 1
  fi
}

ensure_missing() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    echo "[FATAL] target log already exists: ${path}" >&2
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
  local launcher_log="${LAUNCHER_DIR}/${queue_name}.launcher.log"

  nohup bash "${script_path}" > "${launcher_log}" 2>&1 < /dev/null &
  echo "$!" > "${PID_DIR}/${queue_name}.pid"
  echo "[LAUNCHED] ${queue_name} pid=$(cat "${PID_DIR}/${queue_name}.pid")"
}

ensure_missing "${STATIC_DIR}/S03_badnet_fedtrimmed.log"
ensure_missing "${STATIC_DIR}/S07_blend_fedtrimmed.log"
ensure_missing "${STATIC_DIR}/S11_sig_fedtrimmed.log"

ensure_gpus

gen_dataset_if_missing generate_Cifar10_badnet.py "${BADNET_DATASET}" generate_badnet.log
gen_dataset_if_missing generate_Cifar10_blend.py "${BLEND_DATASET}" generate_blend.log \
  --blend-alpha "${BLEND_ALPHA}"
gen_dataset_if_missing generate_Cifar10_sig.py "${SIG_DATASET}" generate_sig.log \
  --sig-delta "${SIG_DELTA}" \
  --sig-f "${SIG_F}" \
  --sig-label-mode dirty

cat > "${QUEUE_DIR}/gpu_badnet.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${SRC}"
"${PY}" -u main.py \
  -dev cuda -did "${GPU_BADNET}" \
  -data "${BADNET_DATASET}" \
  -m "${BASE_MODEL}" -algo FedTrimmed \
  -ncl 10 -nc "${NUM_CLIENTS}" -jr "${JOIN_RATIO}" -lbs "${BATCH_SIZE}" \
  -lr "${LOCAL_LR}" -ls "${LOCAL_EPOCHS}" \
  -gr "${GLOBAL_ROUNDS}" -eg "${EVAL_GAP}" \
  -go S03_badnet_fedtrimmed \
  --num_adv_clients "${ADV_CLIENTS}" \
  > "${STATIC_DIR}/S03_badnet_fedtrimmed.log" 2>&1
EOF

cat > "${QUEUE_DIR}/gpu_blend.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${SRC}"
"${PY}" -u main.py \
  -dev cuda -did "${GPU_BLEND}" \
  -data "${BLEND_DATASET}" \
  -m "${BASE_MODEL}" -algo FedTrimmed \
  -ncl 10 -nc "${NUM_CLIENTS}" -jr "${JOIN_RATIO}" -lbs "${BATCH_SIZE}" \
  -lr "${LOCAL_LR}" -ls "${LOCAL_EPOCHS}" \
  -gr "${GLOBAL_ROUNDS}" -eg "${EVAL_GAP}" \
  -go S07_blend_fedtrimmed \
  --num_adv_clients "${ADV_CLIENTS}" \
  > "${STATIC_DIR}/S07_blend_fedtrimmed.log" 2>&1
EOF

cat > "${QUEUE_DIR}/gpu_sig.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${SRC}"
"${PY}" -u main.py \
  -dev cuda -did "${GPU_SIG}" \
  -data "${SIG_DATASET}" \
  -m "${BASE_MODEL}" -algo FedTrimmed \
  -ncl 10 -nc "${NUM_CLIENTS}" -jr "${JOIN_RATIO}" -lbs "${BATCH_SIZE}" \
  -lr "${LOCAL_LR}" -ls "${LOCAL_EPOCHS}" \
  -gr "${GLOBAL_ROUNDS}" -eg "${EVAL_GAP}" \
  -go S11_sig_fedtrimmed \
  --num_adv_clients "${ADV_CLIENTS}" \
  > "${STATIC_DIR}/S11_sig_fedtrimmed.log" 2>&1
EOF

chmod +x "${QUEUE_DIR}/gpu_badnet.sh" "${QUEUE_DIR}/gpu_blend.sh" "${QUEUE_DIR}/gpu_sig.sh"

launch_queue gpu_badnet
launch_queue gpu_blend
launch_queue gpu_sig

echo "[READY] trimmed-only static launcher started"
echo "[RUN_ROOT] ${RUN_ROOT}"
echo "[STATIC_DIR] ${STATIC_DIR}"
