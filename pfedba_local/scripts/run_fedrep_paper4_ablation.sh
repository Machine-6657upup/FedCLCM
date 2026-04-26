#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/huangtu/PFL_clean_workspace/root_static/pfedba_local"
PYTHON_BIN="/home/huangtu/miniconda3/envs/torch/bin/python"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${ROOT_DIR}/log/fedrep_paper4_${RUN_TS}"
mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
  --dataset Cifar10
  --model resnet
  --resnet_pretrained 0
  --learning_rate 0.1
  --lr_head 0.1
  --plocal_epochs 1
  --local_epochs 10
  --num_global_iters 400
  --numusers 10
  --batch_size 64
  --attack_start 30
  --attack_method attackall
  --poisoning_per_batch 5
  --defense none
  --per_epoch 1
  --malclient 10
  --times 1
)

launch_one() {
  local gpu="$1"
  local tag="$2"
  shift 2
  local logfile="${LOG_DIR}/${tag}.log"
  local metalen="${LOG_DIR}/${tag}.meta.log"
  {
    echo "[START] ${tag}"
    echo "TIME=$(date '+%F %T')"
    echo "GPU=${gpu}"
    printf 'CMD=%q ' "${PYTHON_BIN}" -u "${ROOT_DIR}/main.py" "${COMMON_ARGS[@]}" "$@"
    echo
  } > "${metalen}"

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" -u "${ROOT_DIR}/main.py" "${COMMON_ARGS[@]}" "$@" > "${logfile}" 2>&1
  ) &
  local pid=$!
  echo "PID=${pid}" >> "${metalen}"
  echo "${tag} GPU=${gpu} PID=${pid} LOG=${logfile}"
}

launch_one 0 cluster_only \
  --algorithm FedRepPFLALP \
  --alp_use_cluster 1 \
  --alp_use_purify 0 \
  --cluster_max_k 4

launch_one 1 inter_only \
  --algorithm FedRepBDPFL \
  --bd_lambda 1.0 \
  --bd_tau 1.0 \
  --bd_gamma 1.0 \
  --bd_use_inter 1 \
  --bd_use_em 0

launch_one 2 pflalp_full \
  --algorithm PFLALP \
  --purify_beta 1500 \
  --purify_rounds 1 \
  --cluster_max_k 4

launch_one 3 bdpfl_full \
  --algorithm BDPFL \
  --bd_lambda 1.0 \
  --bd_tau 1.0 \
  --bd_gamma 1.0 \
  --bd_use_inter 1 \
  --bd_use_em 1

echo "RUN_TS=${RUN_TS}"
echo "LOG_DIR=${LOG_DIR}"
wait
