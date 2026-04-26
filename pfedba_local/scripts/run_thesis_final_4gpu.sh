#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/huangtu/PFL_clean_workspace/root_static/pfedba_local"
PYTHON_BIN="${PYTHON_BIN:-/home/huangtu/miniconda3/envs/torch/bin/python}"
MAIN_PY="${ROOT_DIR}/main.py"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-${ROOT_DIR}/log/thesis_final_${RUN_TS}}"
STATUS_DIR="${STATUS_DIR:-${LOGDIR}/status}"
QUEUE="${QUEUE:-}"
DRY_RUN="${DRY_RUN:-0}"
DETACHED_OK="${DETACHED_OK:-0}"
LAUNCH_MODE="${LAUNCH_MODE:-screen}"
FORCE_TAGS="${FORCE_TAGS:-}"
SKIP_TAGS="${SKIP_TAGS:-}"
RUN_GROUPS="${RUN_GROUPS:-all}"

TIMES="${TIMES:-3}"
SEED="${SEED:-1}"
APPENDIX_TIMES="${APPENDIX_TIMES:-1}"
APPENDIX_SEED="${APPENDIX_SEED:-1}"

mkdir -p "${LOGDIR}" "${STATUS_DIR}"

tag_is_forced() {
  local tag="$1"
  if [[ -z "${FORCE_TAGS}" ]]; then
    return 1
  fi
  local norm_list=",${FORCE_TAGS// /},"
  [[ "${norm_list}" == *",${tag},"* ]]
}

tag_is_skipped() {
  local tag="$1"
  if [[ -z "${SKIP_TAGS}" ]]; then
    return 1
  fi
  local norm_list=",${SKIP_TAGS// /},"
  [[ "${norm_list}" == *",${tag},"* ]]
}

group_is_enabled() {
  local group="$1"
  if [[ "${RUN_GROUPS}" == "all" ]]; then
    return 0
  fi
  local norm_list=",${RUN_GROUPS// /},"
  [[ "${norm_list}" == *",${group},"* ]]
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
  --times "${TIMES}"
  --seed "${SEED}"
)

FEDRT_ARGS=(
  --rt_beta 0.10
  --adv_eps 0.10
  --adv_num_iter 5
  --aug_strength 0.10
)

FEDCLCM_ARGS=(
  --rt_beta 0.20
  --lambda_cl 0.20
  --mask_tau 12.0
  --mask_alpha 0.70
  --adv_eps 0
  --adv_num_iter 0
  --enable_channel_mask 1
  --cosine_gate 0
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
    echo "[FATAL] launcher requires at least 4 visible GPUs." >&2
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
    echo "[FATAL] worker mode must be launched from detached screen/nohup flow." >&2
    return 1
  fi
}

write_manifest() {
  cat > "${LOGDIR}/manifest.txt" <<EOF
Thesis final experiment suite

Goal:
- use the recent validated PFedBA-on-FedRep hyperparameters as the thesis mainline
- keep FedRep as the personalized baseline
- use FedRT (AdvPurge line) and FedCLCM as the two core defense methods
- keep classical FL defenses only as FedAvg-side baselines
- move paper-full reproductions to appendix-style runs

Common args:
  dataset=Cifar10 model=resnet resnet_pretrained=0
  total_clients=100 selected_users_per_round=10
  learning_rate=0.1 lr_head=0.1 plocal_epochs=1
  batch_size=64 attack_start=30 attack_method=attackall
  poisoning_per_batch=1 defense=none per_epoch=1
  malclient=10 malicious_pool_mode=legacy_fixed
  times=${TIMES} seed=${SEED}

Recent validated method hyperparameters:
  FedRT   : rt_beta=0.10 adv_eps=0.10 adv_num_iter=5 aug_strength=0.10
  FedCLCM : rt_beta=0.20 lambda_cl=0.20 mask_tau=12.0 mask_alpha=0.70
             adv_eps=0 adv_num_iter=0 enable_channel_mask=1 cosine_gate=0

Groups:
  main:
    T01 FedAvg defense=none                 LE1 GI1000
    T02 FedAvg defense=mkrum                LE1 GI1000
    T03 FedAvg defense=trim                 LE1 GI1000
    T04 FedRep                              LE1 GI1000
    T05 FedRT                               LE1 GI1000
    T06 FedCLCM                             LE1 GI1000

  epoch:
    T07 FedRep                              LE10 GI400
    T08 FedRT                               LE10 GI400
    T09 FedCLCM                             LE10 GI400

  attackers:
    T10 FedRep   malclient=0                LE1 GI1000
    T11 FedRT    malclient=0                LE1 GI1000
    T12 FedCLCM  malclient=0                LE1 GI1000
    T13 FedRep   malclient=1                LE1 GI1000
    T14 FedRT    malclient=1                LE1 GI1000
    T15 FedCLCM  malclient=1                LE1 GI1000
    T16 FedRep   malclient=50               LE1 GI1000
    T17 FedRT    malclient=50               LE1 GI1000
    T18 FedCLCM  malclient=50               LE1 GI1000

  ppb:
    T19 FedRep   poisoning_per_batch=3      LE1 GI1000
    T20 FedRT    poisoning_per_batch=3      LE1 GI1000
    T21 FedCLCM  poisoning_per_batch=3      LE1 GI1000
    T22 FedRep   poisoning_per_batch=5      LE1 GI1000
    T23 FedRT    poisoning_per_batch=5      LE1 GI1000
    T24 FedCLCM  poisoning_per_batch=5      LE1 GI1000

  ablation:
    T25 FedCLCM no_trim      rt_beta=0.00
    T26 FedCLCM weak_trim    rt_beta=0.10
    T27 FedCLCM no_cl        lambda_cl=0.00
    T28 FedCLCM no_mask      enable_channel_mask=0

  appendix:
    T29 PFLALP full paper-aligned
        attack_start=0 selection_strategy=fixed_malicious_mix fixed_malicious_per_round=3
        malclient_id_mode=seeded_pool malclient=30 local_epochs=1 num_global_iters=100
        purify_beta=1500 purify_rounds=1 cluster_max_k=4 mal_local_epoch=6 mal_learning_rate=0.05
        times=${APPENDIX_TIMES} seed=${APPENDIX_SEED}
    T30 BDPFL full paper-aligned
        attack_start=0 selection_strategy=fixed_malicious_mix fixed_malicious_per_round=3
        malclient_id_mode=seeded_pool malclient=3 local_epochs=20 num_global_iters=1000
        lr_decay=0.99 lr_decay_step=10 bd_lambda=1 bd_tau=1 bd_gamma=1 bd_use_inter=1 bd_use_em=1
        times=${APPENDIX_TIMES} seed=${APPENDIX_SEED}

Groups selector:
  RUN_GROUPS=all
  RUN_GROUPS=main,epoch
  RUN_GROUPS=attackers
  RUN_GROUPS=ppb,ablation
  RUN_GROUPS=appendix
EOF
}

run_one() {
  local gpu="$1"
  local tag="$2"
  local group="$3"
  local algorithm="$4"
  local local_epochs="$5"
  local num_global_iters="$6"
  shift 6

  local log_file="${LOGDIR}/${tag}.log"
  local meta_file="${LOGDIR}/${tag}.meta.log"
  local running_file="${STATUS_DIR}/${tag}.running"
  local ok_file="${STATUS_DIR}/${tag}.ok"
  local fail_file="${STATUS_DIR}/${tag}.fail"
  local mpl_dir="/tmp/mpl_${RUN_TS}_${tag}"

  if ! group_is_enabled "${group}"; then
    echo "[SKIP_GROUP] ${tag} group=${group} RUN_GROUPS=${RUN_GROUPS}" >&2
    return 0
  fi

  if [[ -f "${ok_file}" ]] && ! tag_is_forced "${tag}"; then
    echo "[SKIP_OK] ${tag} already completed successfully." >&2
    return 0
  fi

  if tag_is_skipped "${tag}"; then
    echo "[SKIP_TAG] ${tag} skipped by SKIP_TAGS=${SKIP_TAGS}" >&2
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
    echo "GROUP=${group}"
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
      run_or_flag 0 T01 main FedAvg 1 1000
      run_or_flag 0 T05 main FedRT 1 1000 "${FEDRT_ARGS[@]}"
      run_or_flag 0 T10 attackers FedRep 1 1000 --malclient 0
      run_or_flag 0 T14 attackers FedRT 1 1000 --malclient 1 "${FEDRT_ARGS[@]}"
      run_or_flag 0 T18 attackers FedCLCM 1 1000 --malclient 50 "${FEDCLCM_ARGS[@]}"
      run_or_flag 0 T23 ppb FedRT 1 1000 --poisoning_per_batch 5 "${FEDRT_ARGS[@]}"
      run_or_flag 0 T27 ablation FedCLCM 1 1000 --lambda_cl 0.00 --rt_beta 0.20 --mask_tau 12.0 --mask_alpha 0.70 --adv_eps 0 --adv_num_iter 0 --enable_channel_mask 1 --cosine_gate 0
      ;;
    gpu1)
      run_or_flag 1 T02 main FedAvg 1 1000 --defense mkrum
      run_or_flag 1 T06 main FedCLCM 1 1000 "${FEDCLCM_ARGS[@]}"
      run_or_flag 1 T07 epoch FedRep 10 400
      run_or_flag 1 T09 epoch FedCLCM 10 400 "${FEDCLCM_ARGS[@]}"
      run_or_flag 1 T11 attackers FedRT 1 1000 --malclient 0 "${FEDRT_ARGS[@]}"
      run_or_flag 1 T15 attackers FedCLCM 1 1000 --malclient 1 "${FEDCLCM_ARGS[@]}"
      run_or_flag 1 T19 ppb FedRep 1 1000 --poisoning_per_batch 3
      run_or_flag 1 T24 ppb FedCLCM 1 1000 --poisoning_per_batch 5 "${FEDCLCM_ARGS[@]}"
      run_or_flag 1 T29 appendix PFLALP 1 100 --attack_start 0 --selection_strategy fixed_malicious_mix --fixed_malicious_per_round 3 --malclient_id_mode seeded_pool --malclient 30 --purify_beta 1500 --purify_rounds 1 --cluster_max_k 4 --mal_local_epoch 6 --mal_learning_rate 0.05 --times "${APPENDIX_TIMES}" --seed "${APPENDIX_SEED}"
      ;;
    gpu2)
      run_or_flag 2 T03 main FedAvg 1 1000 --defense trim
      run_or_flag 2 T04 main FedRep 1 1000
      run_or_flag 2 T08 epoch FedRT 10 400 "${FEDRT_ARGS[@]}"
      run_or_flag 2 T12 attackers FedCLCM 1 1000 --malclient 0 "${FEDCLCM_ARGS[@]}"
      run_or_flag 2 T16 attackers FedRep 1 1000 --malclient 50
      run_or_flag 2 T20 ppb FedRT 1 1000 --poisoning_per_batch 3 "${FEDRT_ARGS[@]}"
      run_or_flag 2 T25 ablation FedCLCM 1 1000 --rt_beta 0.00 --lambda_cl 0.20 --mask_tau 12.0 --mask_alpha 0.70 --adv_eps 0 --adv_num_iter 0 --enable_channel_mask 1 --cosine_gate 0
      run_or_flag 2 T28 ablation FedCLCM 1 1000 --rt_beta 0.20 --lambda_cl 0.20 --mask_tau 12.0 --mask_alpha 0.70 --adv_eps 0 --adv_num_iter 0 --enable_channel_mask 0 --cosine_gate 0
      ;;
    gpu3)
      run_or_flag 3 T13 attackers FedRep 1 1000 --malclient 1
      run_or_flag 3 T17 attackers FedRT 1 1000 --malclient 50 "${FEDRT_ARGS[@]}"
      run_or_flag 3 T21 ppb FedCLCM 1 1000 --poisoning_per_batch 3 "${FEDCLCM_ARGS[@]}"
      run_or_flag 3 T22 ppb FedRep 1 1000 --poisoning_per_batch 5
      run_or_flag 3 T26 ablation FedCLCM 1 1000 --rt_beta 0.10 --lambda_cl 0.20 --mask_tau 12.0 --mask_alpha 0.70 --adv_eps 0 --adv_num_iter 0 --enable_channel_mask 1 --cosine_gate 0
      run_or_flag 3 T30 appendix BDPFL 20 1000 --attack_start 0 --selection_strategy fixed_malicious_mix --fixed_malicious_per_round 3 --malclient_id_mode seeded_pool --malclient 3 --lr_decay 0.99 --lr_decay_step 10 --bd_lambda 1.0 --bd_tau 1.0 --bd_gamma 1.0 --bd_use_inter 1 --bd_use_em 1 --times "${APPENDIX_TIMES}" --seed "${APPENDIX_SEED}"
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
    echo "SKIP_TAGS=${SKIP_TAGS}"
    echo "RUN_GROUPS=${RUN_GROUPS}"
    echo "TIMES=${TIMES}"
    echo "SEED=${SEED}"
    echo "APPENDIX_TIMES=${APPENDIX_TIMES}"
    echo "APPENDIX_SEED=${APPENDIX_SEED}"
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
      local screen_name="thesis_${RUN_TS}_${q}"
      screen -dmS "${screen_name}" bash -lc "export RUN_TS='${RUN_TS}'; export LOGDIR='${LOGDIR}'; export STATUS_DIR='${STATUS_DIR}'; export QUEUE='${q}'; export DRY_RUN='${DRY_RUN}'; export FORCE_TAGS='${FORCE_TAGS}'; export SKIP_TAGS='${SKIP_TAGS}'; export RUN_GROUPS='${RUN_GROUPS}'; export TIMES='${TIMES}'; export SEED='${SEED}'; export APPENDIX_TIMES='${APPENDIX_TIMES}'; export APPENDIX_SEED='${APPENDIX_SEED}'; export DETACHED_OK=1; bash '${0}' > '${launcher_log}' 2>&1"
      echo "${q} SCREEN=${screen_name} LOG=${launcher_log}" | tee -a "${LOGDIR}/launcher.meta.log"
    else
      nohup env RUN_TS="${RUN_TS}" LOGDIR="${LOGDIR}" STATUS_DIR="${STATUS_DIR}" QUEUE="${q}" DRY_RUN="${DRY_RUN}" FORCE_TAGS="${FORCE_TAGS}" SKIP_TAGS="${SKIP_TAGS}" RUN_GROUPS="${RUN_GROUPS}" TIMES="${TIMES}" SEED="${SEED}" APPENDIX_TIMES="${APPENDIX_TIMES}" APPENDIX_SEED="${APPENDIX_SEED}" DETACHED_OK=1 bash "${0}" > "${launcher_log}" 2>&1 &
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
