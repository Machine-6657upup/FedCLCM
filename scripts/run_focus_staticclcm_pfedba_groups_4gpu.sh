#!/usr/bin/env bash
set -euo pipefail

ROOT_STATIC="/home/huangtu/PFL_clean_workspace/root_static"
PFEDBA_ROOT="${ROOT_STATIC}/pfedba_local"
PFEDBA_MAIN="${PFEDBA_ROOT}/main.py"
STATIC_ROOT="${ROOT_STATIC}/src"
STATIC_MAIN="${STATIC_ROOT}/main.py"
PYTHON_BIN="${PYTHON_BIN:-/home/huangtu/miniconda3/envs/torch/bin/python}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-focus_staticclcm_pfedba_${RUN_TS}}"
RUN_ROOT="${RUN_ROOT:-${ROOT_STATIC}/thesis_formal_logs/${RUN_NAME}}"
STATUS_DIR="${STATUS_DIR:-${RUN_ROOT}/status}"
META_DIR="${META_DIR:-${RUN_ROOT}/meta}"
LAUNCHER_DIR="${LAUNCHER_DIR:-${RUN_ROOT}/launcher}"
QUEUE="${QUEUE:-}"
DETACHED_OK="${DETACHED_OK:-0}"
DRY_RUN="${DRY_RUN:-0}"
LAUNCH_MODE="${LAUNCH_MODE:-screen}"

STATIC_FINAL_ROOT="${STATIC_FINAL_ROOT:-${ROOT_STATIC}/thesis_formal_logs/final_static_main_20260424}"
STATIC_LOG_DIR="${STATIC_LOG_DIR:-${STATIC_FINAL_ROOT}/static}"
PFEDBA_FINAL_ROOT="${PFEDBA_FINAL_ROOT:-${ROOT_STATIC}/thesis_formal_logs/final_pfedba_main_20260424}"
PFEDBA_BATCH_DIR="${PFEDBA_BATCH_DIR:-${PFEDBA_FINAL_ROOT}/${RUN_NAME}}"

BADNET_DATASET="Cifar10_dir0.5_bdoor0.2_nclient_100_badnet_adv10"
BLEND_DATASET="Cifar10_dir0.5_bdoor0.2_nclient_100_blend_adv10"
SIG_DATASET="Cifar10_dir0.5_bdoor0.2_nclient_100_sig_adv10"

mkdir -p "${RUN_ROOT}" "${STATUS_DIR}" "${META_DIR}" "${LAUNCHER_DIR}" "${STATIC_LOG_DIR}" "${PFEDBA_BATCH_DIR}"

PFEDBA_COMMON_ARGS=(
  --dataset Cifar10
  --model resnet
  --resnet_pretrained 0
  --learning_rate 0.1
  --lr_head 0.1
  --plocal_epochs 1
  --numusers 10
  --batch_size 64
  --attack_start 30
  --attack_method attackall
  --poisoning_per_batch 1
  --defense none
  --per_epoch 1
  --malclient 10
  --times 3
  --seed 1
  --eval_gap 10
  --personalized_eval_gap 0
)

PFEDBA_FEDRT_ARGS=(
  --rt_beta 0.10
  --adv_eps 0.10
  --adv_num_iter 5
  --aug_strength 0.10
)

PFEDBA_FEDCLCM_ARGS=(
  --rt_beta 0.20
  --lambda_cl 0.20
  --mask_tau 12.0
  --mask_alpha 0.70
  --adv_eps 0
  --adv_num_iter 0
  --enable_channel_mask 1
  --cosine_gate 0
)

STATIC_COMMON_ARGS=(
  -dev cuda
  -m ResNet18
  -ncl 10
  -nc 100
  -jr 0.1
  -lbs 64
  -lr 0.1
  -ls 1
  -gr 600
  -eg 10
  --num_adv_clients 10
)

STATIC_FEDCLCM_ARGS=(
  -lr_head 0.1
  -pls 1
  --rt_beta 0.2
  --lambda_cl 0.20
  --aug_strength 0.1
  --adv_eps 0
  --adv_num_iter 0
  --mask_tau 12.0
  --mask_alpha 0.70
)

require_detached_worker() {
  if [[ "${DETACHED_OK}" != "1" ]]; then
    echo "[FATAL] worker mode must be launched through detached screen/nohup." >&2
    return 1
  fi
}

require_launcher_gpus() {
  local count
  count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
  echo "gpu_count=${count}"
  if [[ -z "${count}" || "${count}" -lt 4 ]]; then
    echo "[FATAL] launcher requires at least 4 visible GPUs." >&2
    return 1
  fi
}

require_worker_gpu() {
  local gpu="$1"
  local info
  set +e
  info="$(nvidia-smi -i "${gpu}" --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null)"
  local rc=$?
  set -e
  printf '%s\n' "${info}"
  if [[ "${rc}" -ne 0 || -z "${info}" ]]; then
    echo "[FATAL] worker logical gpu ${gpu} is not queryable via nvidia-smi." >&2
    return 1
  fi
}

require_static_datasets() {
  local miss=0
  local dataset
  for dataset in "${BADNET_DATASET}" "${BLEND_DATASET}" "${SIG_DATASET}"; do
    if [[ ! -f "${STATIC_ROOT}/dataset/${dataset}/config.json" ]]; then
      echo "[FATAL] missing static dataset ${dataset}" >&2
      miss=1
    fi
  done
  if [[ "${miss}" != "0" ]]; then
    return 1
  fi
}

ensure_static_targets_missing() {
  local target
  for target in \
    "${STATIC_LOG_DIR}/S04_badnet_fedclcm.log" \
    "${STATIC_LOG_DIR}/S08_blend_fedclcm.log" \
    "${STATIC_LOG_DIR}/S12_sig_fedclcm.log"; do
    if [[ -e "${target}" ]]; then
      echo "[FATAL] static target log already exists: ${target}" >&2
      return 1
    fi
  done
}

write_manifest() {
  cat > "${RUN_ROOT}/manifest.txt" <<EOF
Focused thesis batch

Goal:
- rerun static FedCLCM under lr_head=0.1 for badnet/blend/sig
- run PFedBA missing main baselines FedAvg / FedAvg+mkrum / FedAvg+trim
- run PFedBA attackers / FedCLCM ablation groups

Static targets:
- S04_badnet_fedclcm  dataset=${BADNET_DATASET}
- S08_blend_fedclcm   dataset=${BLEND_DATASET}
- S12_sig_fedclcm     dataset=${SIG_DATASET}
- static args: model=ResNet18 join_ratio=0.1 lr=0.1 lr_head=0.1 local_epochs=1 plocal_epochs=1
               global_rounds=600 eval_gap=10 num_adv_clients=10
               rt_beta=0.2 lambda_cl=0.20 aug_strength=0.1 adv_eps=0 adv_num_iter=0
               mask_tau=12.0 mask_alpha=0.70

PFedBA targets:
- main missing: T01 T02 T03
- attackers: T10 T11 T12 T13 T14 T15 T16 T17 T18
- ablation: T25 T26 T27 T28
- PFedBA args: dataset=Cifar10 model=resnet resnet_pretrained=0
               learning_rate=0.1 lr_head=0.1 plocal_epochs=1
               total_users=100 selected_users_per_round=10
               batch_size=64 attack_start=30 attack_method=attackall
               poisoning_per_batch=1 defense=none per_epoch=1
               eval_gap=10 personalized_eval_gap=0
               times=3 seed=1 malclient=10

Queue layout:
- gpu0: S04_badnet_fedclcm, T01, T10, T14, T18
- gpu1: S08_blend_fedclcm, T02, T11, T15
- gpu2: S12_sig_fedclcm, T03, T12, T16, T25
- gpu3: T13, T17, T26, T27, T28

Output roots:
- static logs: ${STATIC_LOG_DIR}
- pfedba logs: ${PFEDBA_BATCH_DIR}
EOF
}

run_task() {
  local gpu="$1"
  local tag="$2"
  local workdir="$3"
  local log_file="$4"
  local meta_file="$5"
  shift 5

  local running_file="${STATUS_DIR}/${tag}.running"
  local ok_file="${STATUS_DIR}/${tag}.ok"
  local fail_file="${STATUS_DIR}/${tag}.fail"
  local mpl_dir="/tmp/mpl_${RUN_TS}_${tag}"

  rm -f "${ok_file}" "${fail_file}"
  : > "${running_file}"
  mkdir -p "${mpl_dir}" "$(dirname "${log_file}")" "$(dirname "${meta_file}")"

  {
    echo "=================================================="
    echo "[START] ${tag}"
    echo "TIME=$(date '+%F %T')"
    echo "GPU=${gpu}"
    echo "WORKDIR=${workdir}"
    echo "LOG=${log_file}"
    printf 'CMD=%q ' "$@"
    echo
    echo "=================================================="
  } > "${meta_file}" 2>&1

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] ${tag}" >> "${meta_file}"
    rm -f "${running_file}"
    : > "${ok_file}"
    return 0
  fi

  local gpu_info
  if ! gpu_info="$(require_worker_gpu "${gpu}" 2>&1)"; then
    {
      echo "[GPU_CHECK]"
      printf '%s\n' "${gpu_info}"
      echo "=================================================="
      echo "[END] ${tag} RC=97 TIME=$(date '+%F %T')"
      echo "=================================================="
    } >> "${meta_file}"
    rm -f "${running_file}"
    : > "${fail_file}"
    return 1
  fi

  {
    echo "[GPU_CHECK]"
    printf '%s\n' "${gpu_info}"
    echo "=================================================="
  } >> "${meta_file}"

  set +e
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHONUNBUFFERED=1
    export MPLCONFIGDIR="${mpl_dir}"
    cd "${workdir}"
    "$@" > "${log_file}" 2>&1
  )
  local rc=$?
  set -e

  rm -f "${running_file}"
  if [[ "${rc}" -eq 0 ]]; then
    : > "${ok_file}"
  else
    : > "${fail_file}"
  fi

  {
    echo "[END] ${tag} RC=${rc} TIME=$(date '+%F %T')"
    echo "=================================================="
  } >> "${meta_file}"

  return "${rc}"
}

run_queue() {
  require_detached_worker
  require_static_datasets
  local queue_rc=0

  run_or_flag() {
    if ! run_task "$@"; then
      queue_rc=1
    fi
  }

  case "${QUEUE}" in
    gpu0)
      run_or_flag 0 S04_badnet_fedclcm "${STATIC_ROOT}" \
        "${STATIC_LOG_DIR}/S04_badnet_fedclcm.log" "${META_DIR}/S04_badnet_fedclcm.meta.log" \
        "${PYTHON_BIN}" -u "${STATIC_MAIN}" "${STATIC_COMMON_ARGS[@]}" -did 0 \
        -data "${BADNET_DATASET}" -algo FedCLCM -go S04_badnet_fedclcm "${STATIC_FEDCLCM_ARGS[@]}"
      run_or_flag 0 T01 "${PFEDBA_ROOT}" \
        "${PFEDBA_BATCH_DIR}/T01.log" "${META_DIR}/T01.meta.log" \
        "${PYTHON_BIN}" -u "${PFEDBA_MAIN}" "${PFEDBA_COMMON_ARGS[@]}" \
        --algorithm FedAvg --local_epochs 1 --num_global_iters 1000
      run_or_flag 0 T10 "${PFEDBA_ROOT}" \
        "${PFEDBA_BATCH_DIR}/T10.log" "${META_DIR}/T10.meta.log" \
        "${PYTHON_BIN}" -u "${PFEDBA_MAIN}" "${PFEDBA_COMMON_ARGS[@]}" \
        --algorithm FedRep --local_epochs 1 --num_global_iters 1000 --malclient 0
      run_or_flag 0 T14 "${PFEDBA_ROOT}" \
        "${PFEDBA_BATCH_DIR}/T14.log" "${META_DIR}/T14.meta.log" \
        "${PYTHON_BIN}" -u "${PFEDBA_MAIN}" "${PFEDBA_COMMON_ARGS[@]}" \
        --algorithm FedRT --local_epochs 1 --num_global_iters 1000 --malclient 1 "${PFEDBA_FEDRT_ARGS[@]}"
      run_or_flag 0 T18 "${PFEDBA_ROOT}" \
        "${PFEDBA_BATCH_DIR}/T18.log" "${META_DIR}/T18.meta.log" \
        "${PYTHON_BIN}" -u "${PFEDBA_MAIN}" "${PFEDBA_COMMON_ARGS[@]}" \
        --algorithm FedCLCM --local_epochs 1 --num_global_iters 1000 --malclient 50 "${PFEDBA_FEDCLCM_ARGS[@]}"
      ;;
    gpu1)
      run_or_flag 1 S08_blend_fedclcm "${STATIC_ROOT}" \
        "${STATIC_LOG_DIR}/S08_blend_fedclcm.log" "${META_DIR}/S08_blend_fedclcm.meta.log" \
        "${PYTHON_BIN}" -u "${STATIC_MAIN}" "${STATIC_COMMON_ARGS[@]}" -did 1 \
        -data "${BLEND_DATASET}" -algo FedCLCM -go S08_blend_fedclcm "${STATIC_FEDCLCM_ARGS[@]}"
      run_or_flag 1 T02 "${PFEDBA_ROOT}" \
        "${PFEDBA_BATCH_DIR}/T02.log" "${META_DIR}/T02.meta.log" \
        "${PYTHON_BIN}" -u "${PFEDBA_MAIN}" "${PFEDBA_COMMON_ARGS[@]}" \
        --algorithm FedAvg --local_epochs 1 --num_global_iters 1000 --defense mkrum
      run_or_flag 1 T11 "${PFEDBA_ROOT}" \
        "${PFEDBA_BATCH_DIR}/T11.log" "${META_DIR}/T11.meta.log" \
        "${PYTHON_BIN}" -u "${PFEDBA_MAIN}" "${PFEDBA_COMMON_ARGS[@]}" \
        --algorithm FedRT --local_epochs 1 --num_global_iters 1000 --malclient 0 "${PFEDBA_FEDRT_ARGS[@]}"
      run_or_flag 1 T15 "${PFEDBA_ROOT}" \
        "${PFEDBA_BATCH_DIR}/T15.log" "${META_DIR}/T15.meta.log" \
        "${PYTHON_BIN}" -u "${PFEDBA_MAIN}" "${PFEDBA_COMMON_ARGS[@]}" \
        --algorithm FedCLCM --local_epochs 1 --num_global_iters 1000 --malclient 1 "${PFEDBA_FEDCLCM_ARGS[@]}"
      ;;
    gpu2)
      run_or_flag 2 S12_sig_fedclcm "${STATIC_ROOT}" \
        "${STATIC_LOG_DIR}/S12_sig_fedclcm.log" "${META_DIR}/S12_sig_fedclcm.meta.log" \
        "${PYTHON_BIN}" -u "${STATIC_MAIN}" "${STATIC_COMMON_ARGS[@]}" -did 2 \
        -data "${SIG_DATASET}" -algo FedCLCM -go S12_sig_fedclcm "${STATIC_FEDCLCM_ARGS[@]}"
      run_or_flag 2 T03 "${PFEDBA_ROOT}" \
        "${PFEDBA_BATCH_DIR}/T03.log" "${META_DIR}/T03.meta.log" \
        "${PYTHON_BIN}" -u "${PFEDBA_MAIN}" "${PFEDBA_COMMON_ARGS[@]}" \
        --algorithm FedAvg --local_epochs 1 --num_global_iters 1000 --defense trim
      run_or_flag 2 T12 "${PFEDBA_ROOT}" \
        "${PFEDBA_BATCH_DIR}/T12.log" "${META_DIR}/T12.meta.log" \
        "${PYTHON_BIN}" -u "${PFEDBA_MAIN}" "${PFEDBA_COMMON_ARGS[@]}" \
        --algorithm FedCLCM --local_epochs 1 --num_global_iters 1000 --malclient 0 "${PFEDBA_FEDCLCM_ARGS[@]}"
      run_or_flag 2 T16 "${PFEDBA_ROOT}" \
        "${PFEDBA_BATCH_DIR}/T16.log" "${META_DIR}/T16.meta.log" \
        "${PYTHON_BIN}" -u "${PFEDBA_MAIN}" "${PFEDBA_COMMON_ARGS[@]}" \
        --algorithm FedRep --local_epochs 1 --num_global_iters 1000 --malclient 50
      run_or_flag 2 T25 "${PFEDBA_ROOT}" \
        "${PFEDBA_BATCH_DIR}/T25.log" "${META_DIR}/T25.meta.log" \
        "${PYTHON_BIN}" -u "${PFEDBA_MAIN}" "${PFEDBA_COMMON_ARGS[@]}" \
        --algorithm FedCLCM --local_epochs 1 --num_global_iters 1000 \
        --rt_beta 0.00 --lambda_cl 0.20 --mask_tau 12.0 --mask_alpha 0.70 \
        --adv_eps 0 --adv_num_iter 0 --enable_channel_mask 1 --cosine_gate 0
      ;;
    gpu3)
      run_or_flag 3 T13 "${PFEDBA_ROOT}" \
        "${PFEDBA_BATCH_DIR}/T13.log" "${META_DIR}/T13.meta.log" \
        "${PYTHON_BIN}" -u "${PFEDBA_MAIN}" "${PFEDBA_COMMON_ARGS[@]}" \
        --algorithm FedRep --local_epochs 1 --num_global_iters 1000 --malclient 1
      run_or_flag 3 T17 "${PFEDBA_ROOT}" \
        "${PFEDBA_BATCH_DIR}/T17.log" "${META_DIR}/T17.meta.log" \
        "${PYTHON_BIN}" -u "${PFEDBA_MAIN}" "${PFEDBA_COMMON_ARGS[@]}" \
        --algorithm FedRT --local_epochs 1 --num_global_iters 1000 --malclient 50 "${PFEDBA_FEDRT_ARGS[@]}"
      run_or_flag 3 T26 "${PFEDBA_ROOT}" \
        "${PFEDBA_BATCH_DIR}/T26.log" "${META_DIR}/T26.meta.log" \
        "${PYTHON_BIN}" -u "${PFEDBA_MAIN}" "${PFEDBA_COMMON_ARGS[@]}" \
        --algorithm FedCLCM --local_epochs 1 --num_global_iters 1000 \
        --rt_beta 0.10 --lambda_cl 0.20 --mask_tau 12.0 --mask_alpha 0.70 \
        --adv_eps 0 --adv_num_iter 0 --enable_channel_mask 1 --cosine_gate 0
      run_or_flag 3 T27 "${PFEDBA_ROOT}" \
        "${PFEDBA_BATCH_DIR}/T27.log" "${META_DIR}/T27.meta.log" \
        "${PYTHON_BIN}" -u "${PFEDBA_MAIN}" "${PFEDBA_COMMON_ARGS[@]}" \
        --algorithm FedCLCM --local_epochs 1 --num_global_iters 1000 \
        --rt_beta 0.20 --lambda_cl 0.00 --mask_tau 12.0 --mask_alpha 0.70 \
        --adv_eps 0 --adv_num_iter 0 --enable_channel_mask 1 --cosine_gate 0
      run_or_flag 3 T28 "${PFEDBA_ROOT}" \
        "${PFEDBA_BATCH_DIR}/T28.log" "${META_DIR}/T28.meta.log" \
        "${PYTHON_BIN}" -u "${PFEDBA_MAIN}" "${PFEDBA_COMMON_ARGS[@]}" \
        --algorithm FedCLCM --local_epochs 1 --num_global_iters 1000 \
        --rt_beta 0.20 --lambda_cl 0.20 --mask_tau 12.0 --mask_alpha 0.70 \
        --adv_eps 0 --adv_num_iter 0 --enable_channel_mask 0 --cosine_gate 0
      ;;
    *)
      echo "unknown or empty QUEUE=${QUEUE}" >&2
      exit 2
      ;;
  esac

  return "${queue_rc}"
}

launch_all() {
  write_manifest
  require_static_datasets
  ensure_static_targets_missing

  {
    echo "RUN_TS=${RUN_TS}"
    echo "RUN_NAME=${RUN_NAME}"
    echo "RUN_ROOT=${RUN_ROOT}"
    echo "STATIC_LOG_DIR=${STATIC_LOG_DIR}"
    echo "PFEDBA_BATCH_DIR=${PFEDBA_BATCH_DIR}"
    echo "START_TIME=$(date '+%F %T')"
    echo "LAUNCH_MODE=${LAUNCH_MODE}"
    echo "DRY_RUN=${DRY_RUN}"
  } > "${RUN_ROOT}/launcher.meta.log"

  if [[ "${DRY_RUN}" != "1" ]]; then
    {
      echo "[LAUNCHER_GPU_CHECK]"
      require_launcher_gpus
      echo "=================================================="
    } >> "${RUN_ROOT}/launcher.meta.log"
  fi

  local q
  for q in gpu0 gpu1 gpu2 gpu3; do
    local launcher_log="${LAUNCHER_DIR}/${q}.launcher.log"
    if [[ "${LAUNCH_MODE}" == "screen" ]]; then
      local screen_name="${RUN_NAME}_${q}"
      screen -dmS "${screen_name}" bash -lc "export RUN_TS='${RUN_TS}'; export RUN_NAME='${RUN_NAME}'; export RUN_ROOT='${RUN_ROOT}'; export STATUS_DIR='${STATUS_DIR}'; export META_DIR='${META_DIR}'; export LAUNCHER_DIR='${LAUNCHER_DIR}'; export STATIC_FINAL_ROOT='${STATIC_FINAL_ROOT}'; export STATIC_LOG_DIR='${STATIC_LOG_DIR}'; export PFEDBA_FINAL_ROOT='${PFEDBA_FINAL_ROOT}'; export PFEDBA_BATCH_DIR='${PFEDBA_BATCH_DIR}'; export QUEUE='${q}'; export DRY_RUN='${DRY_RUN}'; export DETACHED_OK=1; bash '${0}' > '${launcher_log}' 2>&1"
      echo "${q} SCREEN=${screen_name} LOG=${launcher_log}" | tee -a "${RUN_ROOT}/launcher.meta.log"
    else
      nohup env RUN_TS="${RUN_TS}" RUN_NAME="${RUN_NAME}" RUN_ROOT="${RUN_ROOT}" STATUS_DIR="${STATUS_DIR}" META_DIR="${META_DIR}" LAUNCHER_DIR="${LAUNCHER_DIR}" STATIC_FINAL_ROOT="${STATIC_FINAL_ROOT}" STATIC_LOG_DIR="${STATIC_LOG_DIR}" PFEDBA_FINAL_ROOT="${PFEDBA_FINAL_ROOT}" PFEDBA_BATCH_DIR="${PFEDBA_BATCH_DIR}" QUEUE="${q}" DRY_RUN="${DRY_RUN}" DETACHED_OK=1 bash "${0}" > "${launcher_log}" 2>&1 &
      local pid=$!
      echo "${q} PID=${pid} LOG=${launcher_log}" | tee -a "${RUN_ROOT}/launcher.meta.log"
    fi
  done
}

if [[ -z "${QUEUE}" ]]; then
  launch_all
else
  run_queue
fi
