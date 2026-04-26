#!/usr/bin/env bash
set -euo pipefail
cd "/home/huangtu/PFL_Backdoor_Defense/PFedBA"
export CUDA_VISIBLE_DEVICES="1"
echo "[WORKER_START] gpu1a GPU=1 TIME=$(date '+%F %T')"
rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024101/status/F03.ok" "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024101/status/F03.fail"
touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024101/status/F03.running"
{
  echo "=================================================="
  echo "[START] F03"
  echo "GPU=1"
  echo "WORKER=gpu1a"
  echo "TIME=$(date '+%F %T')"
  echo "LOG=/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024101/F03_gpu1_gpu1a.log"
  echo "CMD=python -u main.py --dataset Cifar10 --model resnet --resnet_pretrained 0 --algorithm FedRT --learning_rate 0.1 --lr_head 0.1 --plocal_epochs 1 --local_epochs 20 --num_global_iters 150 --numusers 10 --batch_size 64 --attack_start 30 --attack_method attackall --poisoning_per_batch 5 --defense none --per_epoch 1 --malclient 10 --times 1 --rt_beta 0.00 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10 "
  echo "=================================================="
} >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024101/F03_gpu1_gpu1a.log" 2>&1

if python -u main.py --dataset Cifar10 --model resnet --resnet_pretrained 0 --algorithm FedRT --learning_rate 0.1 --lr_head 0.1 --plocal_epochs 1 --local_epochs 20 --num_global_iters 150 --numusers 10 --batch_size 64 --attack_start 30 --attack_method attackall --poisoning_per_batch 5 --defense none --per_epoch 1 --malclient 10 --times 1 --rt_beta 0.00 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10  >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024101/F03_gpu1_gpu1a.log" 2>&1; then
  {
    echo "=================================================="
    echo "[END] F03"
    echo "TIME=$(date '+%F %T')"
    echo "=================================================="
  } >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024101/F03_gpu1_gpu1a.log" 2>&1
  rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024101/status/F03.running"
  touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024101/status/F03.ok"
else
  rc=$?
  {
    echo "=================================================="
    echo "[FAIL] F03"
    echo "TIME=$(date '+%F %T')"
    echo "EXIT_CODE=${rc}"
    echo "=================================================="
  } >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024101/F03_gpu1_gpu1a.log" 2>&1
  rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024101/status/F03.running"
  touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024101/status/F03.fail"
  exit "${rc}"
fi

rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024101/status/F11.ok" "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024101/status/F11.fail"
touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024101/status/F11.running"
{
  echo "=================================================="
  echo "[START] F11"
  echo "GPU=1"
  echo "WORKER=gpu1a"
  echo "TIME=$(date '+%F %T')"
  echo "LOG=/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024101/F11_gpu1_gpu1a.log"
  echo "CMD=python -u main.py --dataset Cifar10 --model resnet --resnet_pretrained 0 --algorithm FedRT --learning_rate 0.1 --lr_head 0.1 --plocal_epochs 1 --local_epochs 20 --num_global_iters 150 --numusers 10 --batch_size 64 --attack_start 30 --attack_method attackall --poisoning_per_batch 5 --defense none --per_epoch 1 --malclient 10 --times 1 --rt_beta 0.12 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10 "
  echo "=================================================="
} >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024101/F11_gpu1_gpu1a.log" 2>&1

if python -u main.py --dataset Cifar10 --model resnet --resnet_pretrained 0 --algorithm FedRT --learning_rate 0.1 --lr_head 0.1 --plocal_epochs 1 --local_epochs 20 --num_global_iters 150 --numusers 10 --batch_size 64 --attack_start 30 --attack_method attackall --poisoning_per_batch 5 --defense none --per_epoch 1 --malclient 10 --times 1 --rt_beta 0.12 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10  >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024101/F11_gpu1_gpu1a.log" 2>&1; then
  {
    echo "=================================================="
    echo "[END] F11"
    echo "TIME=$(date '+%F %T')"
    echo "=================================================="
  } >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024101/F11_gpu1_gpu1a.log" 2>&1
  rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024101/status/F11.running"
  touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024101/status/F11.ok"
else
  rc=$?
  {
    echo "=================================================="
    echo "[FAIL] F11"
    echo "TIME=$(date '+%F %T')"
    echo "EXIT_CODE=${rc}"
    echo "=================================================="
  } >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024101/F11_gpu1_gpu1a.log" 2>&1
  rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024101/status/F11.running"
  touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024101/status/F11.fail"
  exit "${rc}"
fi

