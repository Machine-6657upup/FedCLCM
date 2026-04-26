#!/usr/bin/env bash
# =============================================================================
# PFedBA 下 FedRT 的 16 组激进夜跑脚本
# -----------------------------------------------------------------------------
# 设计:
#   - 4 张 GPU
#   - 每张 GPU 同时跑 2 个 worker
#   - 每个 worker 顺序跑 2 组实验
#   - 总计 8 个并发 worker, 16 组实验
#
# 说明:
#   - 旧版采用“顶层 bash + 后台函数 + tee 管道”方式，宿主机上出现过只写
#     [START] 头但没有真正拉起 python 子进程的情况
#   - 新版改成“为每个 worker 生成独立脚本，再用 nohup 单独拉起”，让 worker
#     与顶层启动器解耦，适合关终端夜跑
#
# 启动:
#   cd /home/huangtu/PFL_Backdoor_Defense/PFedBA
#   nohup bash scripts/run_fedrt_pfedba_16_dual_per_gpu.sh > log/fedrt_pfedba16_launcher.out 2>&1 &
#
# 可选:
#   GPUS=0,1,2,3 BATCH_SIZE=32 RUN_TS=20260418_dual bash scripts/run_fedrt_pfedba_16_dual_per_gpu.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PFEDBA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PFEDBA_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/home/huangtu/miniconda3/envs/torch/bin/python}"
MAIN_PY="${PFEDBA_ROOT}/main.py"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${PFEDBA_ROOT}/log/fedrt_pfedba16_dual_${RUN_TS}}"
STATUS_DIR="${LOG_ROOT}/status"
WORKER_DIR="${LOG_ROOT}/workers"
PID_DIR="${LOG_ROOT}/pids"
mkdir -p "${LOG_ROOT}" "${STATUS_DIR}" "${WORKER_DIR}" "${PID_DIR}"

GPUS_CSV="${GPUS:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
if [[ "${#GPUS[@]}" -ne 4 ]]; then
  echo "GPUS must contain exactly 4 GPU ids, got: ${GPUS_CSV}"
  exit 1
fi

BATCH_SIZE="${BATCH_SIZE:-64}"

COMMON=(
  --dataset Cifar10
  --model resnet
  --resnet_pretrained 0
  --algorithm FedRT
  --learning_rate 0.1
  --lr_head 0.1
  --plocal_epochs 1
  --local_epochs 20
  --num_global_iters 150
  --numusers 10
  --batch_size "${BATCH_SIZE}"
  --attack_start 30
  --attack_method attackall
  --poisoning_per_batch 5
  --defense none
  --per_epoch 1
  --malclient 10
  --times 1
)

build_args() {
  local exp_id="$1"
  local -n out_ref="$2"
  out_ref=("${COMMON[@]}")
  case "${exp_id}" in
    F01) out_ref+=(--rt_beta 0.10 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10) ;;
    F02) out_ref+=(--rt_beta 0.10 --adv_eps 0.00 --adv_num_iter 0 --aug_strength 0.10) ;;
    F03) out_ref+=(--rt_beta 0.00 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10) ;;
    F04) out_ref+=(--rt_beta 0.10 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.00) ;;
    F05) out_ref+=(--rt_beta 0.10 --adv_eps 0.05 --adv_num_iter 3 --aug_strength 0.10) ;;
    F06) out_ref+=(--rt_beta 0.10 --adv_eps 0.08 --adv_num_iter 5 --aug_strength 0.10) ;;
    F07) out_ref+=(--rt_beta 0.10 --adv_eps 0.12 --adv_num_iter 5 --aug_strength 0.10) ;;
    F08) out_ref+=(--rt_beta 0.10 --adv_eps 0.15 --adv_num_iter 7 --aug_strength 0.10) ;;
    F09) out_ref+=(--rt_beta 0.05 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10) ;;
    F10) out_ref+=(--rt_beta 0.08 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10) ;;
    F11) out_ref+=(--rt_beta 0.12 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10) ;;
    F12) out_ref+=(--rt_beta 0.15 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10) ;;
    F13) out_ref+=(--lr_head 0.05 --rt_beta 0.10 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10) ;;
    F14) out_ref+=(--lr_head 0.02 --rt_beta 0.10 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10) ;;
    F15) out_ref+=(--learning_rate 0.05 --lr_head 0.05 --rt_beta 0.10 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10) ;;
    F16) out_ref+=(--lr_head 0.05 --rt_beta 0.15 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10) ;;
    *)
      echo "Unknown experiment id: ${exp_id}" >&2
      exit 1
      ;;
  esac
}

quote_cmd() {
  local arg
  for arg in "$@"; do
    printf '%q ' "${arg}"
  done
}

emit_run_block() {
  local exp_id="$1"
  local gpu_id="$2"
  local worker_id="$3"
  shift 3
  local log_file="${LOG_ROOT}/${exp_id}_gpu${gpu_id}_${worker_id}.log"
  local ok_file="${STATUS_DIR}/${exp_id}.ok"
  local fail_file="${STATUS_DIR}/${exp_id}.fail"
  local run_file="${STATUS_DIR}/${exp_id}.running"
  local cmd_quoted
  cmd_quoted="$(quote_cmd "${PYTHON_BIN}" -u "${MAIN_PY}" "$@")"

  cat <<EOF
rm -f "${ok_file}" "${fail_file}"
touch "${run_file}"
{
  echo "=================================================="
  echo "[START] ${exp_id}"
  echo "GPU=${gpu_id}"
  echo "WORKER=${worker_id}"
  echo "TIME=\$(date '+%F %T')"
  echo "LOG=${log_file}"
  echo "CMD=${cmd_quoted}"
  echo "=================================================="
} >> "${log_file}" 2>&1

if ${cmd_quoted} >> "${log_file}" 2>&1; then
  {
    echo "=================================================="
    echo "[END] ${exp_id}"
    echo "TIME=\$(date '+%F %T')"
    echo "=================================================="
  } >> "${log_file}" 2>&1
  rm -f "${run_file}"
  touch "${ok_file}"
else
  rc=\$?
  {
    echo "=================================================="
    echo "[FAIL] ${exp_id}"
    echo "TIME=\$(date '+%F %T')"
    echo "EXIT_CODE=\${rc}"
    echo "=================================================="
  } >> "${log_file}" 2>&1
  rm -f "${run_file}"
  touch "${fail_file}"
  exit "\${rc}"
fi

EOF
}

emit_worker_script() {
  local gpu_id="$1"
  local worker_id="$2"
  local exp_a="$3"
  local exp_b="$4"
  local script_path="${WORKER_DIR}/${worker_id}.sh"
  local args_a=()
  local args_b=()

  build_args "${exp_a}" args_a
  build_args "${exp_b}" args_b

  cat > "${script_path}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${PFEDBA_ROOT}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONUNBUFFERED=1
echo "[WORKER_START] ${worker_id} GPU=${gpu_id} TIME=\$(date '+%F %T')"
EOF
  emit_run_block "${exp_a}" "${gpu_id}" "${worker_id}" "${args_a[@]}" >> "${script_path}"
  emit_run_block "${exp_b}" "${gpu_id}" "${worker_id}" "${args_b[@]}" >> "${script_path}"
  cat >> "${script_path}" <<EOF
echo "[WORKER_END] ${worker_id} GPU=${gpu_id} TIME=\$(date '+%F %T')"
EOF
  chmod +x "${script_path}"
}

launch_worker() {
  local gpu_id="$1"
  local worker_id="$2"
  local script_path="${WORKER_DIR}/${worker_id}.sh"
  local worker_log="${LOG_ROOT}/${worker_id}.worker.log"
  local pid_file="${PID_DIR}/${worker_id}.pid"
  nohup bash "${script_path}" > "${worker_log}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${pid_file}"
  echo "[LAUNCHED] ${worker_id} GPU=${gpu_id} PID=${pid} SCRIPT=${script_path} LOG=${worker_log}"
}

echo "PFEDBA_ROOT=${PFEDBA_ROOT}"
echo "RUN_TS=${RUN_TS}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "GPUS=${GPUS_CSV}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "WORKER_DIR=${WORKER_DIR}"
echo "PID_DIR=${PID_DIR}"
echo "STATUS_DIR=${STATUS_DIR}"

emit_worker_script "${GPUS[0]}" gpu0a F01 F09
emit_worker_script "${GPUS[0]}" gpu0b F02 F10
emit_worker_script "${GPUS[1]}" gpu1a F03 F11
emit_worker_script "${GPUS[1]}" gpu1b F04 F12
emit_worker_script "${GPUS[2]}" gpu2a F05 F13
emit_worker_script "${GPUS[2]}" gpu2b F06 F14
emit_worker_script "${GPUS[3]}" gpu3a F07 F15
emit_worker_script "${GPUS[3]}" gpu3b F08 F16

launch_worker "${GPUS[0]}" gpu0a
launch_worker "${GPUS[0]}" gpu0b
launch_worker "${GPUS[1]}" gpu1a
launch_worker "${GPUS[1]}" gpu1b
launch_worker "${GPUS[2]}" gpu2a
launch_worker "${GPUS[2]}" gpu2b
launch_worker "${GPUS[3]}" gpu3a
launch_worker "${GPUS[3]}" gpu3b

echo "=================================================="
echo "Detached 8 workers at $(date '+%F %T')"
echo "Use the files under ${PID_DIR} and ${STATUS_DIR} to monitor progress."
echo "=================================================="
