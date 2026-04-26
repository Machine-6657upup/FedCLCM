#!/usr/bin/env bash
set -euo pipefail

# README-aligned FedRep attack run for FashionMnist
ROOT_DIR="/home/huangtu/PFL_Backdoor_Defense/PFedBA"
LOG_DIR="${ROOT_DIR}/log/fedrep_nohup"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/fedrep_FashionMnist_${RUN_TS}.log"

mkdir -p "${LOG_DIR}"

nohup python -u "${ROOT_DIR}/main.py" \
  --dataset FashionMnist \
  --algorithm FedRep \
  --learning_rate 0.1 \
  --numusers 10 \
  --local_epochs 20 \
  --num_global_iters 150 \
  --batch_size 64 \
  --malclient 10 \
  --attack_method attackall \
  --attack_start 30 \
  --poisoning_per_batch 16 \
  --defense none \
  --per_epoch 1 \
  --plocal_epochs 1 \
  --lr_head 0.1 \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Started. PID=${PID}"
echo "Log file: ${LOG_FILE}"
echo "Tail: tail -f \"${LOG_FILE}\""
echo "Stop: kill ${PID}"
