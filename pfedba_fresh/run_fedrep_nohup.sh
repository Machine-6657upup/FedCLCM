#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# PFedBA (original repo) - FedRep nohup launcher
#
# Usage:
#   bash run_fedrep_nohup.sh                   # default: Cifar10
#   DATASET=FashionMnist bash run_fedrep_nohup.sh
#   GPU=1 DATASET=Cifar10 ROUNDS=400 bash run_fedrep_nohup.sh
###############################################################################

ROOT_DIR="/home/huangtu/PFL_Backdoor_Defense/PFedBA"
LOG_DIR="${ROOT_DIR}/log/fedrep_nohup"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

GPU="${GPU:-0}"
DATASET="${DATASET:-Cifar10}"            # Cifar10 / FashionMnist / Mnist
ROUNDS="${ROUNDS:-200}"                  # quick default, you can set 800
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
NUM_USERS_PER_ROUND="${NUM_USERS_PER_ROUND:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-0.05}"
MAL_CLIENTS="${MAL_CLIENTS:-5}"
ATTACK_START="${ATTACK_START:-10}"
POISON_PER_BATCH="${POISON_PER_BATCH:-5}"  # original code uses count
DEFENSE="${DEFENSE:-none}"                 # none / mkrum / trim
PLOCAL_EPOCHS="${PLOCAL_EPOCHS:-1}"
LR_HEAD="${LR_HEAD:-0.1}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/fedrep_${DATASET}_${RUN_TS}.log"

echo "============================================================"
echo "Launching PFedBA FedRep with nohup"
echo "GPU=${GPU} DATASET=${DATASET} ROUNDS=${ROUNDS}"
echo "LOG=${LOG_FILE}"
echo "============================================================"

nohup env CUDA_VISIBLE_DEVICES="${GPU}" \
python -u "${ROOT_DIR}/main.py" \
  --dataset "${DATASET}" \
  --algorithm FedRep \
  --num_global_iters "${ROUNDS}" \
  --local_epochs "${LOCAL_EPOCHS}" \
  --numusers "${NUM_USERS_PER_ROUND}" \
  --batch_size "${BATCH_SIZE}" \
  --learning_rate "${LR}" \
  --plocal_epochs "${PLOCAL_EPOCHS}" \
  --lr_head "${LR_HEAD}" \
  --malclient "${MAL_CLIENTS}" \
  --attack_start "${ATTACK_START}" \
  --poisoning_per_batch "${POISON_PER_BATCH}" \
  --attack_method attackall \
  --defense "${DEFENSE}" \
  --times 1 \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Started. PID=${PID}"
echo "Tail log:"
echo "  tail -f \"${LOG_FILE}\""
echo "Stop:"
echo "  kill ${PID}"
