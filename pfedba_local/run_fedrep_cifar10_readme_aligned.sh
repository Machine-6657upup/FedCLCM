#!/usr/bin/env bash
set -euo pipefail

# Final CIFAR10 FedRep runner (Chapter-4 aligned, nohup mode)
# Usage:
#   bash run_fedrep_cifar10_readme_aligned.sh
#   GPU_ID=1 bash run_fedrep_cifar10_readme_aligned.sh

ROOT_DIR="/home/huangtu/PFL_Backdoor_Defense/PFedBA"
LOG_DIR="${ROOT_DIR}/log/fedrep_nohup"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/fedrep_Cifar10_readme_${RUN_TS}.log"
GPU_ID="${GPU_ID:-0}"

mkdir -p "${LOG_DIR}"

# Avoid duplicated runs with same entry command.
if pgrep -af "python -u ${ROOT_DIR}/main.py --dataset Cifar10 --algorithm FedRep" >/dev/null; then
  echo "Detected an existing CIFAR10 FedRep run."
  echo "Please stop it first if you want a fresh run:"
  echo "  pgrep -af \"python -u ${ROOT_DIR}/main.py --dataset Cifar10 --algorithm FedRep\""
  echo "  pkill -f \"python -u ${ROOT_DIR}/main.py --dataset Cifar10 --algorithm FedRep\""
  exit 1
fi

nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" python -u "${ROOT_DIR}/main.py" \
  --dataset Cifar10 \
  --algorithm FedRep \
  --model resnet \
  --learning_rate 0.1 \
  --numusers 10 \
  --local_epochs 20 \
  --num_global_iters 150 \
  --batch_size 64 \
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
echo "GPU_ID=${GPU_ID}"
echo "Log file: ${LOG_FILE}"
echo "Tail: tail -f \"${LOG_FILE}\""
echo "Stop: kill ${PID}"
