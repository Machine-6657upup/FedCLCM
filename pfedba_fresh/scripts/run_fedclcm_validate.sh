#!/usr/bin/env bash
# =============================================================================
# PFedBA 下 FedCLCM 验证脚本
# -----------------------------------------------------------------------------
# 训练侧（与 PFedBA README / 常见 FedRep 强基线一致，便于和仓库内 FedRep 日志对比）:
#   --learning_rate 0.1 --lr_head 0.1 --plocal_epochs 1 --local_epochs 20
#   --num_global_iters 150 --numusers 10 --batch_size 64
#   --attack_start 30 --attack_method attackall --defense none
#   --poisoning_per_batch 5 --per_epoch 1 --malclient 10
#
# FedCLCM 防御/客户端侧超参来源（根目录 thesis 脚本，与 01/07/10 等一致）:
#   - Fashion-MNIST: scripts/01_main_results_basic.sh 中 fmnist_fedclcm 行
#       rt_beta=0.18, lambda_cl=0.05, aug_strength=0.03, mask_tau=8.0, mask_alpha=0.80
#       （原配置含 trim_high_layers=fc1 与分层 trim；PFedBA to_fedrep_model 的 base 参数名
#        多为 Sequential 数字前缀，与 fc1 字符串前缀不一定匹配，故本脚本仅用全局 rt_beta）
#   - CIFAR-10: scripts/10_fedclcm_badnet_training_hyperparam_sweep.sh / 07_heterogeneity_study.sh
#       rt_beta=0.05, lambda_cl=0.20, aug_strength=0.10, mask_tau=12.0, mask_alpha=0.70, adv 关闭
#
# 用法:
#   cd PFedBA && bash scripts/run_fedclcm_validate.sh
#   LOG_ROOT=/path/to/logs NUM_GLOBAL_ITERS=300 GPUS=0 bash scripts/run_fedclcm_validate.sh
#   RUN_FEDREP_BASELINE=1 bash scripts/run_fedclcm_validate.sh   # 同配置再跑 FedRep 对照
#
# PFedBA/main.py 当前还能直接跑什么（算法分支已实现）:
#   - 算法: FedAvg, FedProx, FedRep, FedCLCM（parser 里列出的 Ditto/SCAFFOLD/FedBN 未接分支，勿选）
#   - 防御: none | mkrum | trim（FedCLCM 聚合自带鲁棒，一般仍用 --defense none）
#   - 数据集: Mnist | FashionMnist | Cifar10
#   - 模型: Cifar10 下 --model resnet 与 --resnet_pretrained 0/1；FMnist 固定 FMnistNet
#   - 攻击: 仅 attackall 流程（与论文代码一致）
#
# 可在此基础上自行扩展的实验（改本脚本或命令行即可）:
#   - 换 poisoning_per_batch、attack_start、numusers、num_global_iters
#   - FedAvg/FedProx + defense mkrum/trim 作基线对比
#   - Cifar10 --model cnn 使用 CifarNet（FedRep/FedCLCM 需 to_fedrep_model 支持的结构）
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PFEDBA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PFEDBA_ROOT}"

LOG_ROOT="${LOG_ROOT:-${PFEDBA_ROOT}/log/fedclcm_validate}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_ROOT}"

NUM_GLOBAL_ITERS="${NUM_GLOBAL_ITERS:-150}"
GPUS="${GPUS:-0}"
RUN_FEDREP_BASELINE="${RUN_FEDREP_BASELINE:-0}"

# ---------- PFedBA FedRep 常用训练超参（与 README / fedrep_nohup 一类设置对齐）----------
TRAIN=(
  --learning_rate 0.1
  --lr_head 0.1
  --plocal_epochs 1
  --local_epochs 20
  --num_global_iters "${NUM_GLOBAL_ITERS}"
  --numusers 10
  --batch_size 64
  --attack_start 30
  --attack_method attackall
  --defense none
  --poisoning_per_batch 5
  --per_epoch 1
  --malclient 10
  --times 1
)

# ---------- FedCLCM：Fashion-MNIST（01_main_results_basic fmnist_fedclcm）----------
FM_CLCM=(
  --rt_beta 0.18
  --lambda_cl 0.05
  --aug_strength 0.03
  --adv_eps 0.0
  --adv_num_iter 0
  --mask_tau 8.0
  --mask_alpha 0.80
  --enable_channel_mask 1
  --cosine_gate 0
)

# ---------- FedCLCM：CIFAR-10（10_fedclcm_badnet_training / 07_heterogeneity_study）----------
CIFAR_CLCM=(
  --rt_beta 0.05
  --lambda_cl 0.20
  --aug_strength 0.10
  --adv_eps 0.0
  --adv_num_iter 0
  --mask_tau 12.0
  --mask_alpha 0.70
  --enable_channel_mask 1
  --cosine_gate 0
)

run_one() {
  local name="$1"
  shift
  local log="${LOG_ROOT}/${name}_${RUN_TS}.log"
  echo "=========================================="
  echo "[RUN] ${name}"
  echo "  log: ${log}"
  echo "=========================================="
  python -u main.py "$@" 2>&1 | tee "${log}"
}

# ----- 主实验：FedCLCM -----
run_one "fedclcm_fmnist" \
  --dataset FashionMnist --model cnn --algorithm FedCLCM \
  "${TRAIN[@]}" "${FM_CLCM[@]}"

run_one "fedclcm_cifar10_resnet" \
  --dataset Cifar10 --model resnet --resnet_pretrained 0 --algorithm FedCLCM \
  "${TRAIN[@]}" "${CIFAR_CLCM[@]}"

run_one "fedclcm_cifar10_resnet_pretrained" \
  --dataset Cifar10 --model resnet --resnet_pretrained 1 --algorithm FedCLCM \
  "${TRAIN[@]}" "${CIFAR_CLCM[@]}"

# ----- 可选：同超参 FedRep 对照 -----
if [[ "${RUN_FEDREP_BASELINE}" == "1" ]]; then
  run_one "fedrep_fmnist" \
    --dataset FashionMnist --model cnn --algorithm FedRep \
    "${TRAIN[@]}"
  run_one "fedrep_cifar10_resnet" \
    --dataset Cifar10 --model resnet --resnet_pretrained 0 --algorithm FedRep \
    "${TRAIN[@]}"
  run_one "fedrep_cifar10_resnet_pretrained" \
    --dataset Cifar10 --model resnet --resnet_pretrained 1 --algorithm FedRep \
    "${TRAIN[@]}"
fi

echo "All runs finished. Logs under: ${LOG_ROOT}"
