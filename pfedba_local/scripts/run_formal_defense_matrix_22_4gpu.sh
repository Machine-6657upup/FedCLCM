#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/huangtu/PFL_clean_workspace/root_static/pfedba_local"
PYTHON_BIN="${PYTHON_BIN:-/home/huangtu/miniconda3/envs/torch/bin/python}"
MAIN_PY="${ROOT_DIR}/main.py"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-${ROOT_DIR}/log/formal_defense_matrix_22_${RUN_TS}}"
STATUS_DIR="${STATUS_DIR:-${LOGDIR}/status}"
QUEUE="${QUEUE:-}"
DRY_RUN="${DRY_RUN:-0}"
DETACHED_OK="${DETACHED_OK:-0}"
LAUNCH_MODE="${LAUNCH_MODE:-nohup}"
FORCE_TAGS="${FORCE_TAGS:-}"

mkdir -p "${LOGDIR}" "${STATUS_DIR}"

tag_is_forced() {
  local tag="$1"
  if [[ -z "${FORCE_TAGS}" ]]; then
    return 1
  fi
  local norm_list=",${FORCE_TAGS// /},"
  [[ "${norm_list}" == *",${tag},"* ]]
}

COMMON_ARGS=(
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
  --times 1
)

require_launcher_gpus() {
  local check_out
  check_out="$("${PYTHON_BIN}" - <<'PY'
import sys
import torch

ok = torch.cuda.is_available()
count = torch.cuda.device_count() if ok else 0
print(f"cuda_available={ok}")
print(f"device_count={count}")
sys.exit(0 if ok and count >= 4 else 1)
PY
)"
  local rc=$?
  printf '%s\n' "${check_out}"
  if [[ "${rc}" -ne 0 ]]; then
    echo "[FATAL] launcher requires at least 4 visible GPUs; refuse to start in CPU-only / partial-GPU environment." >&2
    return 1
  fi
}

require_worker_gpu() {
  local gpu="$1"
  local check_out
  check_out="$(CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" - <<'PY'
import sys
import torch

ok = torch.cuda.is_available()
count = torch.cuda.device_count() if ok else 0
name = torch.cuda.get_device_name(0) if ok and count >= 1 else "N/A"
print(f"cuda_available={ok}")
print(f"device_count={count}")
print(f"device0={name}")
sys.exit(0 if ok and count == 1 else 1)
PY
)"
  local rc=$?
  printf '%s\n' "${check_out}"
  if [[ "${rc}" -ne 0 ]]; then
    echo "[FATAL] worker queue ${QUEUE} on logical gpu ${gpu} does not have exactly one visible CUDA device." >&2
    return 1
  fi
}

require_detached_worker() {
  if [[ "${DETACHED_OK}" != "1" ]]; then
    echo "[FATAL] worker mode must be launched from detached nohup/screen flow; direct foreground worker execution is forbidden." >&2
    return 1
  fi
}

write_manifest() {
  cat > "${LOGDIR}/manifest.txt" <<'EOF'
Formal matrix: 22 runs
Common args:
  dataset=Cifar10 model=resnet resnet_pretrained=0
  learning_rate=0.1 lr_head=0.1 plocal_epochs=1
  numusers=10 batch_size=64 attack_start=30 attack_method=attackall
  poisoning_per_batch=1 defense=none per_epoch=1 malclient=10 times=1

LE10 / GI400:
  E01 FedRep
  E02 FedRepPFLALP cluster_only: alp_use_cluster=1 alp_use_purify=0 cluster_max_k=4
  E03 FedRepBDPFL inter_only: bd_lambda=1.0 bd_tau=1.0 bd_gamma=1.0 bd_use_inter=1 bd_use_em=0
  E04 PFLALP full: purify_beta=1500 purify_rounds=1 cluster_max_k=4
  E05 BDPFL full: bd_lambda=1.0 bd_tau=1.0 bd_gamma=1.0 bd_use_inter=1 bd_use_em=1
  E06 FedRT: rt_beta=0.10 adv_eps=0.10 adv_num_iter=5 aug_strength=0.10
  E07 FedCLCM: rt_beta=0.20 lambda_cl=0.20 mask_tau=12.0 mask_alpha=0.70 adv_eps=0 adv_num_iter=0 enable_channel_mask=1 cosine_gate=0
  E08 FedRepPFLALP cluster_only_strong: alp_use_cluster=1 alp_use_purify=0 cluster_max_k=6
  E09 FedRepBDPFL inter_only_strong: bd_lambda=1.0 bd_tau=1.0 bd_gamma=2.0 bd_use_inter=1 bd_use_em=0
  E10 PFLALP full_strong: purify_beta=1500 purify_rounds=2 cluster_max_k=4
  E11 BDPFL full_strong: bd_lambda=1.0 bd_tau=1.0 bd_gamma=2.0 bd_use_inter=1 bd_use_em=1

LE1 / GI1000:
  E12 FedRep
  E13 FedRepPFLALP cluster_only: alp_use_cluster=1 alp_use_purify=0 cluster_max_k=4
  E14 FedRepBDPFL inter_only: bd_lambda=1.0 bd_tau=1.0 bd_gamma=1.0 bd_use_inter=1 bd_use_em=0
  E15 PFLALP full: purify_beta=1500 purify_rounds=1 cluster_max_k=4
  E16 BDPFL full: bd_lambda=1.0 bd_tau=1.0 bd_gamma=1.0 bd_use_inter=1 bd_use_em=1
  E17 FedRT: rt_beta=0.10 adv_eps=0.10 adv_num_iter=5 aug_strength=0.10
  E18 FedCLCM: rt_beta=0.20 lambda_cl=0.20 mask_tau=12.0 mask_alpha=0.70 adv_eps=0 adv_num_iter=0 enable_channel_mask=1 cosine_gate=0
  E19 FedRepPFLALP cluster_only_strong: alp_use_cluster=1 alp_use_purify=0 cluster_max_k=6
  E20 FedRepBDPFL inter_only_strong: bd_lambda=1.0 bd_tau=1.0 bd_gamma=2.0 bd_use_inter=1 bd_use_em=0
  E21 PFLALP full_strong: purify_beta=1500 purify_rounds=2 cluster_max_k=4
  E22 BDPFL full_strong: bd_lambda=1.0 bd_tau=1.0 bd_gamma=2.0 bd_use_inter=1 bd_use_em=1

Queue split:
  gpu0: E01 E05 E09 E13 E17 E21
  gpu1: E02 E06 E10 E14 E18 E22
  gpu2: E03 E07 E11 E15 E19
  gpu3: E04 E08 E12 E16 E20
EOF
}

run_one() {
  local gpu="$1"
  local tag="$2"
  local algorithm="$3"
  local local_epochs="$4"
  local num_global_iters="$5"
  shift 5

  local log_file="${LOGDIR}/${tag}.log"
  local meta_file="${LOGDIR}/${tag}.meta.log"
  local running_file="${STATUS_DIR}/${tag}.running"
  local ok_file="${STATUS_DIR}/${tag}.ok"
  local fail_file="${STATUS_DIR}/${tag}.fail"
  local mpl_dir="/tmp/mpl_${RUN_TS}_${tag}"

  if [[ -f "${ok_file}" ]] && ! tag_is_forced "${tag}"; then
    echo "[SKIP_OK] ${tag} already completed successfully; keep existing log/meta." >&2
    return 0
  fi

  rm -f "${ok_file}" "${fail_file}"
  : > "${running_file}"
  mkdir -p "${mpl_dir}"

  {
    echo "=================================================="
    echo "[START] ${tag}"
    echo "TIME=$(date '+%F %T')"
    echo "GPU=${gpu}"
    echo "ALGORITHM=${algorithm}"
    echo "LOCAL_EPOCHS=${local_epochs}"
    echo "NUM_GLOBAL_ITERS=${num_global_iters}"
    echo "LOG=${log_file}"
    printf 'CMD=%q ' "${PYTHON_BIN}" -u "${MAIN_PY}" "${COMMON_ARGS[@]}" --algorithm "${algorithm}" --local_epochs "${local_epochs}" --num_global_iters "${num_global_iters}" "$@"
    echo
    echo "=================================================="
  } > "${meta_file}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] ${tag}" >> "${meta_file}"
    rm -f "${running_file}"
    : > "${ok_file}"
    return 0
  fi

  {
    echo "[GPU_CHECK]"
    require_worker_gpu "${gpu}"
    echo "=================================================="
  } >> "${meta_file}" 2>&1

  set +e
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHONUNBUFFERED=1
    export MPLCONFIGDIR="${mpl_dir}"
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" -u "${MAIN_PY}" \
      "${COMMON_ARGS[@]}" \
      --algorithm "${algorithm}" \
      --local_epochs "${local_epochs}" \
      --num_global_iters "${num_global_iters}" \
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
  local queue_rc=0
  run_or_flag() {
    if ! run_one "$@"; then
      queue_rc=1
    fi
  }
  case "${QUEUE}" in
    gpu0)
      run_or_flag 0 E01 FedRep 10 400
      run_or_flag 0 E05 BDPFL 10 400 --bd_lambda 1.0 --bd_tau 1.0 --bd_gamma 1.0 --bd_use_inter 1 --bd_use_em 1
      run_or_flag 0 E09 FedRepBDPFL 10 400 --bd_lambda 1.0 --bd_tau 1.0 --bd_gamma 2.0 --bd_use_inter 1 --bd_use_em 0
      run_or_flag 0 E13 FedRepPFLALP 1 1000 --alp_use_cluster 1 --alp_use_purify 0 --cluster_max_k 4
      run_or_flag 0 E17 FedRT 1 1000 --rt_beta 0.10 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10
      run_or_flag 0 E21 PFLALP 1 1000 --purify_beta 1500 --purify_rounds 2 --cluster_max_k 4
      ;;
    gpu1)
      run_or_flag 1 E02 FedRepPFLALP 10 400 --alp_use_cluster 1 --alp_use_purify 0 --cluster_max_k 4
      run_or_flag 1 E06 FedRT 10 400 --rt_beta 0.10 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10
      run_or_flag 1 E10 PFLALP 10 400 --purify_beta 1500 --purify_rounds 2 --cluster_max_k 4
      run_or_flag 1 E14 FedRepBDPFL 1 1000 --bd_lambda 1.0 --bd_tau 1.0 --bd_gamma 1.0 --bd_use_inter 1 --bd_use_em 0
      run_or_flag 1 E18 FedCLCM 1 1000 --rt_beta 0.20 --lambda_cl 0.20 --mask_tau 12.0 --mask_alpha 0.70 --adv_eps 0 --adv_num_iter 0 --enable_channel_mask 1 --cosine_gate 0
      run_or_flag 1 E22 BDPFL 1 1000 --bd_lambda 1.0 --bd_tau 1.0 --bd_gamma 2.0 --bd_use_inter 1 --bd_use_em 1
      ;;
    gpu2)
      run_or_flag 2 E03 FedRepBDPFL 10 400 --bd_lambda 1.0 --bd_tau 1.0 --bd_gamma 1.0 --bd_use_inter 1 --bd_use_em 0
      run_or_flag 2 E07 FedCLCM 10 400 --rt_beta 0.20 --lambda_cl 0.20 --mask_tau 12.0 --mask_alpha 0.70 --adv_eps 0 --adv_num_iter 0 --enable_channel_mask 1 --cosine_gate 0
      run_or_flag 2 E11 BDPFL 10 400 --bd_lambda 1.0 --bd_tau 1.0 --bd_gamma 2.0 --bd_use_inter 1 --bd_use_em 1
      run_or_flag 2 E15 PFLALP 1 1000 --purify_beta 1500 --purify_rounds 1 --cluster_max_k 4
      run_or_flag 2 E19 FedRepPFLALP 1 1000 --alp_use_cluster 1 --alp_use_purify 0 --cluster_max_k 6
      ;;
    gpu3)
      run_or_flag 3 E04 PFLALP 10 400 --purify_beta 1500 --purify_rounds 1 --cluster_max_k 4
      run_or_flag 3 E08 FedRepPFLALP 10 400 --alp_use_cluster 1 --alp_use_purify 0 --cluster_max_k 6
      run_or_flag 3 E12 FedRep 1 1000
      run_or_flag 3 E16 BDPFL 1 1000 --bd_lambda 1.0 --bd_tau 1.0 --bd_gamma 1.0 --bd_use_inter 1 --bd_use_em 1
      run_or_flag 3 E20 FedRepBDPFL 1 1000 --bd_lambda 1.0 --bd_tau 1.0 --bd_gamma 2.0 --bd_use_inter 1 --bd_use_em 0
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
  local stale_cleanup_count
  stale_cleanup_count="$(find "${STATUS_DIR}" -maxdepth 1 -type f -name '*.running' | wc -l)"
  if [[ "${stale_cleanup_count}" != "0" ]]; then
    find "${STATUS_DIR}" -maxdepth 1 -type f -name '*.running' -print -delete > "${LOGDIR}/stale_running_cleanup.log"
  fi
  {
    echo "RUN_TS=${RUN_TS}"
    echo "LOGDIR=${LOGDIR}"
    echo "STATUS_DIR=${STATUS_DIR}"
    echo "SCRIPT=${0}"
    echo "START_TIME=$(date '+%F %T')"
    echo "STALE_RUNNING_CLEANUP=${stale_cleanup_count}"
    echo "FORCE_TAGS=${FORCE_TAGS}"
    if [[ "${DRY_RUN}" != "1" ]]; then
      echo "[LAUNCHER_GPU_CHECK]"
      require_launcher_gpus
      echo "=================================================="
    fi
  } > "${LOGDIR}/launcher.meta.log"

  local queues=(gpu0 gpu1 gpu2 gpu3)
  for q in "${queues[@]}"; do
    local launcher_log="${LOGDIR}/${q}.launcher.log"
    if [[ "${LAUNCH_MODE}" == "screen" ]]; then
      local screen_name="fdm22_${RUN_TS}_${q}"
      screen -dmS "${screen_name}" bash -lc "export RUN_TS='${RUN_TS}'; export LOGDIR='${LOGDIR}'; export STATUS_DIR='${STATUS_DIR}'; export QUEUE='${q}'; export DRY_RUN='${DRY_RUN}'; export FORCE_TAGS='${FORCE_TAGS}'; export DETACHED_OK=1; bash '${0}' > '${launcher_log}' 2>&1"
      echo "${q} SCREEN=${screen_name} LOG=${launcher_log}" | tee -a "${LOGDIR}/launcher.meta.log"
    else
      nohup env RUN_TS="${RUN_TS}" LOGDIR="${LOGDIR}" STATUS_DIR="${STATUS_DIR}" QUEUE="${q}" DRY_RUN="${DRY_RUN}" FORCE_TAGS="${FORCE_TAGS}" DETACHED_OK=1 bash "${0}" > "${launcher_log}" 2>&1 &
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
