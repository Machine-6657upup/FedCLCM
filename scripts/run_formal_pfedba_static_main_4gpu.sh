#!/usr/bin/env bash
set -euo pipefail

ROOT_STATIC="/home/huangtu/PFL_clean_workspace/root_static"
PFEDBA_ROOT="${ROOT_STATIC}/pfedba_local"
PFEDBA_MAIN="${PFEDBA_ROOT}/main.py"
STATIC_ROOT="${ROOT_STATIC}/src"
STATIC_MAIN="${STATIC_ROOT}/main.py"
PY_PFEDBA="${PY_PFEDBA:-/home/huangtu/miniconda3/envs/torch/bin/python}"
PY_STATIC="${PY_STATIC:-/home/huangtu/miniconda3/envs/torch/bin/python}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
FORMAL_ROOT="${FORMAL_ROOT:-${ROOT_STATIC}/thesis_formal_logs}"
RUN_NAME="${RUN_NAME:-formal_main_pfedba_static_${RUN_TS}}"
LOGDIR="${LOGDIR:-${FORMAL_ROOT}/${RUN_NAME}}"
STATUS_DIR="${STATUS_DIR:-${LOGDIR}/status}"
QUEUE="${QUEUE:-}"
DETACHED_OK="${DETACHED_OK:-0}"
DRY_RUN="${DRY_RUN:-0}"
LAUNCH_MODE="${LAUNCH_MODE:-nohup}"
TASKS="${TASKS:-}"

mkdir -p "${LOGDIR}/pfedba" "${LOGDIR}/static" "${LOGDIR}/launcher" "${STATUS_DIR}"

BADNET_DATASET="Cifar10_dir0.5_bdoor0.2_nclient_100_badnet_adv10"
BLEND_DATASET="Cifar10_dir0.5_bdoor0.2_nclient_100_blend_adv10"
SIG_DATASET="Cifar10_dir0.5_bdoor0.2_nclient_100_sig_adv10"

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
  --malclient 10
  --times 3
  --seed 1
  --eval_gap 10
  --personalized_eval_gap 0
)

PFEDBA_FEDAVG_EVAL_ARGS=(
  --per_epoch 5
)

PFEDBA_FEDREP_STYLE_EVAL_ARGS=(
  --per_epoch 1
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

STATIC_FEDREP_ARGS=(
  -lr_head 0.1
  -pls 1
)

STATIC_FEDCLCM_ARGS=(
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
    echo "[FATAL] worker mode must be launched via detached nohup/screen flow." >&2
    return 1
  fi
}

require_launcher_gpus() {
  local count
  count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
  echo "gpu_count=${count}"
  if [[ "${count}" -lt 4 ]]; then
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

write_manifest() {
  cat > "${LOGDIR}/manifest.txt" <<EOF
Formal thesis main run

Run name:
  ${RUN_NAME}

Formal log root:
  ${LOGDIR}

PFedBA main setting:
  dataset=Cifar10 model=resnet resnet_pretrained=0
  learning_rate=0.1 lr_head=0.1 plocal_epochs=1
  total_users=100 selected_users_per_round=10
  batch_size=64 attack_start=30 attack_method=attackall
  poisoning_per_batch=1 defense=none eval_gap=10 personalized_eval_gap=0
  malclient=10 times=3 seed=1
  local_epochs=1 global_rounds=1000
  FedAvg/FedAvg+defense final personalized eval: per_epoch=5 (paper-aligned FedAvg-FT on CIFAR-10)
  FedRep/FedRT/FedCLCM final personalized eval: per_epoch=1
  FedRT   : rt_beta=0.10 adv_eps=0.10 adv_num_iter=5 aug_strength=0.10
  FedCLCM : rt_beta=0.20 lambda_cl=0.20 mask_tau=12.0 mask_alpha=0.70
             adv_eps=0 adv_num_iter=0 enable_channel_mask=1 cosine_gate=0

Static main setting:
  dataset family=Cifar10_dir0.5_bdoor0.2_nclient_100_{badnet,blend,sig}_adv10
  model=ResNet18 num_clients=100 join_ratio=0.1 batch_size=64
  learning_rate=0.1 lr_head=0.1 local_epochs=1 plocal_epochs=1
  global_rounds=600 eval_gap=10 num_adv_clients=10
  Blend: blend_alpha=0.2
  SIG  : sig_delta=30/255 sig_f=6 sig_label_mode=dirty
  Static FedCLCM explicit args: rt_beta=0.2 lambda_cl=0.20 aug_strength=0.1
                                adv_eps=0 adv_num_iter=0 mask_tau=12.0 mask_alpha=0.70

Task layout:
  gpu0: P01_fedavg, P05_fedrt, S01_badnet_fedavg, S04_blend_fedavg
  gpu1: P02_fedavg_mkrum, P06_fedclcm, S02_badnet_fedrep, S05_blend_fedrep
  gpu2: P03_fedavg_trim, S03_badnet_fedclcm, S06_blend_fedclcm
  gpu3: P04_fedrep, S07_sig_fedavg, S08_sig_fedrep, S09_sig_fedclcm
EOF
}

run_one() {
  local family="$1"
  local gpu="$2"
  local tag="$3"
  local workdir="$4"
  shift 4

  local family_dir="${LOGDIR}/${family}"
  local log_file="${family_dir}/${tag}.log"
  local meta_file="${family_dir}/${tag}.meta.log"
  local running_file="${STATUS_DIR}/${tag}.running"
  local ok_file="${STATUS_DIR}/${tag}.ok"
  local fail_file="${STATUS_DIR}/${tag}.fail"
  local mpl_dir="/tmp/mpl_${RUN_TS}_${tag}"

  rm -f "${ok_file}" "${fail_file}"
  : > "${running_file}"
  mkdir -p "${mpl_dir}"

  {
    echo "=================================================="
    echo "[START] ${tag}"
    echo "TIME=$(date '+%F %T')"
    echo "GPU=${gpu}"
    echo "FAMILY=${family}"
    echo "WORKDIR=${workdir}"
    echo "LOG=${log_file}"
    printf 'CMD=%q ' "$@"
    echo
    echo "[GPU_CHECK]"
    require_worker_gpu "${gpu}"
    echo "=================================================="
  } > "${meta_file}" 2>&1

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] ${tag}" >> "${meta_file}"
    rm -f "${running_file}"
    : > "${ok_file}"
    return 0
  fi

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

task_selected() {
  local tag="$1"
  if [[ -z "${TASKS}" ]]; then
    return 0
  fi
  [[ ",${TASKS}," == *",${tag},"* ]]
}

run_queue() {
  require_detached_worker
  require_static_datasets
  if [[ ! -f "${LOGDIR}/manifest.txt" ]]; then
    write_manifest
  fi
  local queue_rc=0

  run_or_flag() {
    local tag="$3"
    if ! task_selected "${tag}"; then
      echo "[SKIP] ${tag} not selected by TASKS=${TASKS}"
      return 0
    fi
    if ! run_one "$@"; then
      queue_rc=1
    fi
  }

  case "${QUEUE}" in
    gpu0)
      run_or_flag pfedba 0 P01_fedavg "${PFEDBA_ROOT}" \
        "${PY_PFEDBA}" -u "${PFEDBA_MAIN}" "${PFEDBA_COMMON_ARGS[@]}" \
        --algorithm FedAvg --local_epochs 1 --num_global_iters 1000 "${PFEDBA_FEDAVG_EVAL_ARGS[@]}"
      run_or_flag pfedba 0 P05_fedrt "${PFEDBA_ROOT}" \
        "${PY_PFEDBA}" -u "${PFEDBA_MAIN}" "${PFEDBA_COMMON_ARGS[@]}" \
        --algorithm FedRT --local_epochs 1 --num_global_iters 1000 "${PFEDBA_FEDREP_STYLE_EVAL_ARGS[@]}" "${PFEDBA_FEDRT_ARGS[@]}"
      run_or_flag static 0 S01_badnet_fedavg "${STATIC_ROOT}" \
        "${PY_STATIC}" -u "${STATIC_MAIN}" "${STATIC_COMMON_ARGS[@]}" -did 0 \
        -data "${BADNET_DATASET}" -algo FedAvg -go formal_static_badnet_fedavg
      run_or_flag static 0 S04_blend_fedavg "${STATIC_ROOT}" \
        "${PY_STATIC}" -u "${STATIC_MAIN}" "${STATIC_COMMON_ARGS[@]}" -did 0 \
        -data "${BLEND_DATASET}" -algo FedAvg -go formal_static_blend_fedavg
      ;;
    gpu1)
      run_or_flag pfedba 1 P02_fedavg_mkrum "${PFEDBA_ROOT}" \
        "${PY_PFEDBA}" -u "${PFEDBA_MAIN}" "${PFEDBA_COMMON_ARGS[@]}" \
        --algorithm FedAvg --local_epochs 1 --num_global_iters 1000 "${PFEDBA_FEDAVG_EVAL_ARGS[@]}" --defense mkrum
      run_or_flag pfedba 1 P06_fedclcm "${PFEDBA_ROOT}" \
        "${PY_PFEDBA}" -u "${PFEDBA_MAIN}" "${PFEDBA_COMMON_ARGS[@]}" \
        --algorithm FedCLCM --local_epochs 1 --num_global_iters 1000 "${PFEDBA_FEDREP_STYLE_EVAL_ARGS[@]}" "${PFEDBA_FEDCLCM_ARGS[@]}"
      run_or_flag static 1 S02_badnet_fedrep "${STATIC_ROOT}" \
        "${PY_STATIC}" -u "${STATIC_MAIN}" "${STATIC_COMMON_ARGS[@]}" -did 1 "${STATIC_FEDREP_ARGS[@]}" \
        -data "${BADNET_DATASET}" -algo FedRep -go formal_static_badnet_fedrep
      run_or_flag static 1 S05_blend_fedrep "${STATIC_ROOT}" \
        "${PY_STATIC}" -u "${STATIC_MAIN}" "${STATIC_COMMON_ARGS[@]}" -did 1 "${STATIC_FEDREP_ARGS[@]}" \
        -data "${BLEND_DATASET}" -algo FedRep -go formal_static_blend_fedrep
      ;;
    gpu2)
      run_or_flag pfedba 2 P03_fedavg_trim "${PFEDBA_ROOT}" \
        "${PY_PFEDBA}" -u "${PFEDBA_MAIN}" "${PFEDBA_COMMON_ARGS[@]}" \
        --algorithm FedAvg --local_epochs 1 --num_global_iters 1000 "${PFEDBA_FEDAVG_EVAL_ARGS[@]}" --defense trim
      run_or_flag static 2 S03_badnet_fedclcm "${STATIC_ROOT}" \
        "${PY_STATIC}" -u "${STATIC_MAIN}" "${STATIC_COMMON_ARGS[@]}" -did 2 "${STATIC_FEDCLCM_ARGS[@]}" \
        -data "${BADNET_DATASET}" -algo FedCLCM -go formal_static_badnet_fedclcm
      run_or_flag static 2 S06_blend_fedclcm "${STATIC_ROOT}" \
        "${PY_STATIC}" -u "${STATIC_MAIN}" "${STATIC_COMMON_ARGS[@]}" -did 2 "${STATIC_FEDCLCM_ARGS[@]}" \
        -data "${BLEND_DATASET}" -algo FedCLCM -go formal_static_blend_fedclcm
      ;;
    gpu3)
      run_or_flag pfedba 3 P04_fedrep "${PFEDBA_ROOT}" \
        "${PY_PFEDBA}" -u "${PFEDBA_MAIN}" "${PFEDBA_COMMON_ARGS[@]}" \
        --algorithm FedRep --local_epochs 1 --num_global_iters 1000 "${PFEDBA_FEDREP_STYLE_EVAL_ARGS[@]}"
      run_or_flag static 3 S07_sig_fedavg "${STATIC_ROOT}" \
        "${PY_STATIC}" -u "${STATIC_MAIN}" "${STATIC_COMMON_ARGS[@]}" -did 3 \
        -data "${SIG_DATASET}" -algo FedAvg -go formal_static_sig_fedavg
      run_or_flag static 3 S08_sig_fedrep "${STATIC_ROOT}" \
        "${PY_STATIC}" -u "${STATIC_MAIN}" "${STATIC_COMMON_ARGS[@]}" -did 3 "${STATIC_FEDREP_ARGS[@]}" \
        -data "${SIG_DATASET}" -algo FedRep -go formal_static_sig_fedrep
      run_or_flag static 3 S09_sig_fedclcm "${STATIC_ROOT}" \
        "${PY_STATIC}" -u "${STATIC_MAIN}" "${STATIC_COMMON_ARGS[@]}" -did 3 "${STATIC_FEDCLCM_ARGS[@]}" \
        -data "${SIG_DATASET}" -algo FedCLCM -go formal_static_sig_fedclcm
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
  {
    echo "RUN_TS=${RUN_TS}"
    echo "RUN_NAME=${RUN_NAME}"
    echo "LOGDIR=${LOGDIR}"
    echo "STATUS_DIR=${STATUS_DIR}"
    echo "SCRIPT=${0}"
    echo "START_TIME=$(date '+%F %T')"
    echo "LAUNCH_MODE=${LAUNCH_MODE}"
    echo "DRY_RUN=${DRY_RUN}"
    echo "[LAUNCHER_GPU_CHECK]"
    require_launcher_gpus
    echo "=================================================="
  } > "${LOGDIR}/launcher.meta.log"

  local q
  for q in gpu0 gpu1 gpu2 gpu3; do
    local launcher_log="${LOGDIR}/launcher/${q}.launcher.log"
    if [[ "${LAUNCH_MODE}" == "screen" ]]; then
      local screen_name="${RUN_NAME}_${q}"
      screen -dmS "${screen_name}" bash -lc "export RUN_TS='${RUN_TS}'; export FORMAL_ROOT='${FORMAL_ROOT}'; export RUN_NAME='${RUN_NAME}'; export LOGDIR='${LOGDIR}'; export STATUS_DIR='${STATUS_DIR}'; export QUEUE='${q}'; export DRY_RUN='${DRY_RUN}'; export TASKS='${TASKS}'; export DETACHED_OK=1; bash '${0}' > '${launcher_log}' 2>&1"
      echo "${q} SCREEN=${screen_name} LOG=${launcher_log}" | tee -a "${LOGDIR}/launcher.meta.log"
    else
      nohup env RUN_TS="${RUN_TS}" FORMAL_ROOT="${FORMAL_ROOT}" RUN_NAME="${RUN_NAME}" LOGDIR="${LOGDIR}" STATUS_DIR="${STATUS_DIR}" QUEUE="${q}" DRY_RUN="${DRY_RUN}" TASKS="${TASKS}" DETACHED_OK=1 bash "${0}" > "${launcher_log}" 2>&1 &
      local pid=$!
      echo "${q} PID=${pid} LOG=${launcher_log}" | tee -a "${LOGDIR}/launcher.meta.log"
    fi
  done
}

if [[ -z "${QUEUE}" ]]; then
  launch_all
else
  run_queue
fi
