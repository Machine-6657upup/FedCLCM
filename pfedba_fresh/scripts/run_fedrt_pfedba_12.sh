#!/usr/bin/env bash
# =============================================================================
# PFedBA 下 FedRT 的 12 组正式筛选夜跑脚本
# -----------------------------------------------------------------------------
# 设计目标:
#   - 使用 4 张 GPU 并行
#   - 每张 GPU 顺序跑 3 组，总共 12 组
#   - 只需启动 1 次脚本，即可稳定执行完整队列
#   - 每组实验单独落日志，便于睡醒后直接对比
#
# 推荐启动方式:
#   cd /home/huangtu/PFL_Backdoor_Defense/PFedBA
#   nohup bash scripts/run_fedrt_pfedba_12.sh > log/fedrt_pfedba12_launcher.out 2>&1 &
#
# 可选:
#   GPUS=0,1,2,3 RUN_TS=20260418_overnight bash scripts/run_fedrt_pfedba_12.sh
#
# 输出:
#   - 每组单独日志:   log/fedrt_pfedba12_<RUN_TS>/*.log
#   - 每组状态标记:   log/fedrt_pfedba12_<RUN_TS>/status/*.ok|*.fail
#   - 总控日志:       由外层 nohup 重定向
#
# 明早优先看:
#   - Average Personal Accurancy (k local SGD)
#   - Average Personal ATTACK ALL ASR (k local SGD)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PFEDBA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PFEDBA_ROOT}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${PFEDBA_ROOT}/log/fedrt_pfedba12_${RUN_TS}}"
STATUS_DIR="${LOG_ROOT}/status"
mkdir -p "${LOG_ROOT}" "${STATUS_DIR}"

GPUS_CSV="${GPUS:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
if [[ "${#GPUS[@]}" -ne 4 ]]; then
  echo "GPUS must contain exactly 4 GPU ids, got: ${GPUS_CSV}"
  exit 1
fi

# -----------------------------------------------------------------------------
# 公共训练参数
# -----------------------------------------------------------------------------
# dataset/model:
#   锁定 CIFAR-10 + ResNet，与现有 PFedBA FedRep/FedCLCM 主线一致
# learning_rate:
#   shared base 的本地学习率
# lr_head:
#   个性化 head 的学习率；默认 0.1，仅个别实验改低
# plocal_epochs:
#   每轮 head 更新次数
# local_epochs:
#   每轮 base 更新次数；今晚按你要求保持 20
# num_global_iters:
#   总通信轮数
# numusers:
#   每轮采样客户端数
# batch_size:
#   本地 batch 大小
# attack_start:
#   从第 30 轮开始攻击
# poisoning_per_batch:
#   恶意 batch 内投毒强度，沿用当前 5_1 主线
# defense:
#   外层 defense 固定 none；FedRT 自身 server 端已有 trim
# -----------------------------------------------------------------------------
COMMON=(
  --dataset Cifar10
  --model resnet
  --resnet_pretrained 0
  --algorithm FedRT
  --learning_rate 0.1
  --lr_head 0.1
  --plocal_epochs 1
  --local_epochs 20
  --num_global_iters 150
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

print_exp_help() {
  cat <<'EOF'
Experiment meanings:
  F01 anchor:
    rt_beta=0.10, adv_eps=0.10, adv_num_iter=5, aug_strength=0.10
    主锚点，server trim + benign PGD + benign augmentation 全开
  F02 no_adv:
    去掉 PGD，检查 adversarial purification 是否核心
  F03 no_trim:
    去掉 server trim，检查 trim 是否核心
  F04 no_aug:
    去掉 benign augmentation，检查增强是否有真实贡献
  F05 weak_adv:
    更轻 PGD，测试更温和净化是否更平衡
  F06 strong_adv:
    更强 PGD，测试能否继续压 ASR 或是否伤 clean acc
  F07 weak_trim:
    更轻 trim，测试 0.10 是否过重
  F08 strong_trim:
    更强 trim，测试能否进一步压 PFedBA
  F09 low_head_lr:
    头学习率降到 0.05，测试 head 是否过猛
  F10 very_low_head_lr:
    头学习率降到 0.01，测试更保守个性化头
  F11 low_lr_all:
    base/head 学习率同时降到 0.05，测试整体更稳训练
  F12 strong_candidate:
    rt_beta=0.15 + lr_head=0.05，今晚最值得关注的强候选点
EOF
}

build_args() {
  local exp_id="$1"
  local -n out_ref="$2"
  out_ref=("${COMMON[@]}")
  case "${exp_id}" in
    F01)
      out_ref+=(--rt_beta 0.10 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10)
      ;;
    F02)
      out_ref+=(--rt_beta 0.10 --adv_eps 0.00 --adv_num_iter 0 --aug_strength 0.10)
      ;;
    F03)
      out_ref+=(--rt_beta 0.00 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10)
      ;;
    F04)
      out_ref+=(--rt_beta 0.10 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.00)
      ;;
    F05)
      out_ref+=(--rt_beta 0.10 --adv_eps 0.05 --adv_num_iter 3 --aug_strength 0.10)
      ;;
    F06)
      out_ref+=(--rt_beta 0.10 --adv_eps 0.15 --adv_num_iter 7 --aug_strength 0.10)
      ;;
    F07)
      out_ref+=(--rt_beta 0.05 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10)
      ;;
    F08)
      out_ref+=(--rt_beta 0.15 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10)
      ;;
    F09)
      out_ref+=(--lr_head 0.05 --rt_beta 0.10 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10)
      ;;
    F10)
      out_ref+=(--lr_head 0.01 --rt_beta 0.10 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10)
      ;;
    F11)
      out_ref+=(--learning_rate 0.05 --lr_head 0.05 --rt_beta 0.10 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10)
      ;;
    F12)
      out_ref+=(--lr_head 0.05 --rt_beta 0.15 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10)
      ;;
    *)
      echo "Unknown experiment id: ${exp_id}" >&2
      exit 1
      ;;
  esac
}

run_one() {
  local gpu_id="$1"
  local exp_id="$2"
  shift 2
  local log_file="${LOG_ROOT}/${exp_id}_gpu${gpu_id}.log"
  local ok_file="${STATUS_DIR}/${exp_id}.ok"
  local fail_file="${STATUS_DIR}/${exp_id}.fail"

  rm -f "${ok_file}" "${fail_file}"
  {
    echo "=================================================="
    echo "[START] ${exp_id}"
    echo "GPU=${gpu_id}"
    echo "TIME=$(date '+%F %T')"
    echo "LOG=${log_file}"
    echo "CMD=python -u main.py $*"
    echo "=================================================="
    env CUDA_VISIBLE_DEVICES="${gpu_id}" python -u main.py "$@"
    echo "=================================================="
    echo "[END] ${exp_id}"
    echo "TIME=$(date '+%F %T')"
    echo "=================================================="
  } 2>&1 | tee "${log_file}"

  touch "${ok_file}"
}

worker() {
  local gpu_id="$1"
  shift
  local exp_id
  for exp_id in "$@"; do
    local args=()
    build_args "${exp_id}" args
    if ! run_one "${gpu_id}" "${exp_id}" "${args[@]}"; then
      touch "${STATUS_DIR}/${exp_id}.fail"
      echo "[FAIL] ${exp_id} on GPU ${gpu_id}" >&2
      return 1
    fi
  done
}

echo "PFedBA_ROOT=${PFEDBA_ROOT}"
echo "RUN_TS=${RUN_TS}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "GPUS=${GPUS_CSV}"
print_exp_help

# 3 波 x 4 卡固定队列
# 第一波: 主结论 + 强候选
# 第二波: 拆模块 + trim/head lr 方向
# 第三波: 更细 PGD/head/lr 方向
worker "${GPUS[0]}" F01 F04 F05 &
PID0=$!
worker "${GPUS[1]}" F02 F07 F06 &
PID1=$!
worker "${GPUS[2]}" F03 F08 F10 &
PID2=$!
worker "${GPUS[3]}" F12 F09 F11 &
PID3=$!

FAIL=0
for pid in "${PID0}" "${PID1}" "${PID2}" "${PID3}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done

echo "=================================================="
echo "All workers finished at $(date '+%F %T')"
echo "Status files under: ${STATUS_DIR}"
echo "Logs under: ${LOG_ROOT}"
echo "=================================================="

if [[ "${FAIL}" -ne 0 ]]; then
  echo "At least one worker failed." >&2
  exit 1
fi

