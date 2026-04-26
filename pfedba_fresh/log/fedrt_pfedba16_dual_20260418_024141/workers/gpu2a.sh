#!/usr/bin/env bash
set -euo pipefail
cd "/home/huangtu/PFL_Backdoor_Defense/PFedBA"
export CUDA_VISIBLE_DEVICES="2"
echo "[WORKER_START] gpu2a GPU=2 TIME=$(date '+%F %T')"
rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F05.ok" "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F05.fail"
touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F05.running"
{
  echo "=================================================="
  echo "[START] F05"
  echo "GPU=2"
  echo "WORKER=gpu2a"
  echo "TIME=$(date '+%F %T')"
  echo "LOG=/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/F05_gpu2_gpu2a.log"
  echo "CMD=python -u main.py --dataset Cifar10 --model resnet --resnet_pretrained 0 --algorithm FedRT --learning_rate 0.1 --lr_head 0.1 --plocal_epochs 1 --local_epochs 20 --num_global_iters 150 --numusers 10 --batch_size 64 --attack_start 30 --attack_method attackall --poisoning_per_batch 5 --defense none --per_epoch 1 --malclient 10 --times 1 --rt_beta 0.10 --adv_eps 0.05 --adv_num_iter 3 --aug_strength 0.10 "
  echo "=================================================="
} >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/F05_gpu2_gpu2a.log" 2>&1

if python -u main.py --dataset Cifar10 --model resnet --resnet_pretrained 0 --algorithm FedRT --learning_rate 0.1 --lr_head 0.1 --plocal_epochs 1 --local_epochs 20 --num_global_iters 150 --numusers 10 --batch_size 64 --attack_start 30 --attack_method attackall --poisoning_per_batch 5 --defense none --per_epoch 1 --malclient 10 --times 1 --rt_beta 0.10 --adv_eps 0.05 --adv_num_iter 3 --aug_strength 0.10  >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/F05_gpu2_gpu2a.log" 2>&1; then
  {
    echo "=================================================="
    echo "[END] F05"
    echo "TIME=$(date '+%F %T')"
    echo "=================================================="
  } >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/F05_gpu2_gpu2a.log" 2>&1
  rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F05.running"
  touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F05.ok"
else
  rc=$?
  {
    echo "=================================================="
    echo "[FAIL] F05"
    echo "TIME=$(date '+%F %T')"
    echo "EXIT_CODE=${rc}"
    echo "=================================================="
  } >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/F05_gpu2_gpu2a.log" 2>&1
  rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F05.running"
  touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F05.fail"
  exit "${rc}"
fi

rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F13.ok" "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F13.fail"
touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F13.running"
{
  echo "=================================================="
  echo "[START] F13"
  echo "GPU=2"
  echo "WORKER=gpu2a"
  echo "TIME=$(date '+%F %T')"
  echo "LOG=/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/F13_gpu2_gpu2a.log"
  echo "CMD=python -u main.py --dataset Cifar10 --model resnet --resnet_pretrained 0 --algorithm FedRT --learning_rate 0.1 --lr_head 0.1 --plocal_epochs 1 --local_epochs 20 --num_global_iters 150 --numusers 10 --batch_size 64 --attack_start 30 --attack_method attackall --poisoning_per_batch 5 --defense none --per_epoch 1 --malclient 10 --times 1 --lr_head 0.05 --rt_beta 0.10 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10 "
  echo "=================================================="
} >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/F13_gpu2_gpu2a.log" 2>&1

if python -u main.py --dataset Cifar10 --model resnet --resnet_pretrained 0 --algorithm FedRT --learning_rate 0.1 --lr_head 0.1 --plocal_epochs 1 --local_epochs 20 --num_global_iters 150 --numusers 10 --batch_size 64 --attack_start 30 --attack_method attackall --poisoning_per_batch 5 --defense none --per_epoch 1 --malclient 10 --times 1 --lr_head 0.05 --rt_beta 0.10 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10  >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/F13_gpu2_gpu2a.log" 2>&1; then
  {
    echo "=================================================="
    echo "[END] F13"
    echo "TIME=$(date '+%F %T')"
    echo "=================================================="
  } >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/F13_gpu2_gpu2a.log" 2>&1
  rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F13.running"
  touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F13.ok"
else
  rc=$?
  {
    echo "=================================================="
    echo "[FAIL] F13"
    echo "TIME=$(date '+%F %T')"
    echo "EXIT_CODE=${rc}"
    echo "=================================================="
  } >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/F13_gpu2_gpu2a.log" 2>&1
  rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F13.running"
  touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F13.fail"
  exit "${rc}"
fi

echo "[WORKER_END] gpu2a GPU=2 TIME=$(date '+%F %T')"
