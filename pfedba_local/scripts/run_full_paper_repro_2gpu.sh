#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/huangtu/PFL_clean_workspace/root_static/pfedba_local"
PYTHON_BIN="${PYTHON_BIN:-/home/huangtu/miniconda3/envs/torch/bin/python}"
MAIN_PY="${ROOT_DIR}/main.py"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-${ROOT_DIR}/log/full_paper_repro_${RUN_TS}}"
POISONING_PER_BATCH="${POISONING_PER_BATCH:-1}"

mkdir -p "${LOGDIR}"

COMMON_ARGS=(
  --dataset Cifar10
  --model resnet
  --resnet_pretrained 0
  --batch_size 64
  --numusers 10
  --attack_method attackall
  --attack_start 0
  --per_epoch 1
  --defense none
  --times 1
  --poisoning_per_batch "${POISONING_PER_BATCH}"
  --malclient_id_mode seeded_pool
  --selection_strategy fixed_malicious_mix
)

cat > "${LOGDIR}/README.txt" <<EOF
Paper-aligned full-method launcher

This launcher is separated from the FedRep lite/baseline matrix on purpose.
- PFLALP full:
  round=100, benign local epoch=1, attacker local epoch=6, lr=0.1, attacker lr=0.05
  malicious pool=30, fixed malicious per round=3
- BDPFL full:
  round=1000, local epoch=20, lr=0.1, lr decay=0.99 every 10 rounds
  attackers=3, fixed malicious per round=3

Repo-specific note:
- POISONING_PER_BATCH=${POISONING_PER_BATCH} is a PFedBA attack knob and does not have a direct 1:1 paper counterpart.
EOF

run_one() {
  local gpu="$1"
  local tag="$2"
  shift 2

  local log_file="${LOGDIR}/${tag}.log"
  local meta_file="${LOGDIR}/${tag}.meta.log"
  local mpl_dir="/tmp/mpl_${RUN_TS}_${tag}"

  mkdir -p "${mpl_dir}"

  {
    echo "TIME=$(date '+%F %T')"
    echo "GPU=${gpu}"
    printf 'CMD=%q ' "${PYTHON_BIN}" -u "${MAIN_PY}" "${COMMON_ARGS[@]}" "$@"
    echo
  } > "${meta_file}"

  nohup env CUDA_VISIBLE_DEVICES="${gpu}" MPLCONFIGDIR="${mpl_dir}" \
    "${PYTHON_BIN}" -u "${MAIN_PY}" "${COMMON_ARGS[@]}" "$@" \
    > "${log_file}" 2>&1 &

  echo "${tag} PID=$!" | tee -a "${LOGDIR}/pids.txt"
}

run_one 0 PFLALP_full_paper \
  --algorithm PFLALP \
  --learning_rate 0.1 \
  --num_global_iters 100 \
  --local_epochs 1 \
  --malclient 30 \
  --fixed_malicious_per_round 3 \
  --purify_beta 1500 \
  --purify_rounds 1 \
  --cluster_max_k 4 \
  --mal_local_epoch 6 \
  --mal_learning_rate 0.05

run_one 1 BDPFL_full_paper \
  --algorithm BDPFL \
  --learning_rate 0.1 \
  --num_global_iters 1000 \
  --local_epochs 20 \
  --malclient 3 \
  --fixed_malicious_per_round 3 \
  --lr_decay 0.99 \
  --lr_decay_step 10 \
  --bd_lambda 1.0 \
  --bd_tau 1.0 \
  --bd_gamma 1.0 \
  --bd_use_inter 1 \
  --bd_use_em 1

echo "LOGDIR=${LOGDIR}"
