#!/usr/bin/env bash
set -euo pipefail
cd "/home/huangtu/PFL_Backdoor_Defense/PFedBA"
export CUDA_VISIBLE_DEVICES="2"
echo "[WORKER_START] gpu2b GPU=2 TIME=$(date '+%F %T')"
rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F06.ok" "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F06.fail"
touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F06.running"
{
  echo "=================================================="
  echo "[START] F06"
  echo "GPU=2"
  echo "WORKER=gpu2b"
  echo "TIME=$(date '+%F %T')"
  echo "LOG=/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/F06_gpu2_gpu2b.log"
  echo "CMD=python -u main.py --dataset Cifar10 --model resnet --resnet_pretrained 0 --algorithm FedRT --learning_rate 0.1 --lr_head 0.1 --plocal_epochs 1 --local_epochs 20 --num_global_iters 150 --numusers 10 --batch_size 64 --attack_start 30 --attack_method attackall --poisoning_per_batch 5 --defense none --per_epoch 1 --malclient 10 --times 1 --rt_beta 0.10 --adv_eps 0.08 --adv_num_iter 5 --aug_strength 0.10 "
  echo "=================================================="
} >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/F06_gpu2_gpu2b.log" 2>&1

if python -u main.py --dataset Cifar10 --model resnet --resnet_pretrained 0 --algorithm FedRT --learning_rate 0.1 --lr_head 0.1 --plocal_epochs 1 --local_epochs 20 --num_global_iters 150 --numusers 10 --batch_size 64 --attack_start 30 --attack_method attackall --poisoning_per_batch 5 --defense none --per_epoch 1 --malclient 10 --times 1 --rt_beta 0.10 --adv_eps 0.08 --adv_num_iter 5 --aug_strength 0.10  >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/F06_gpu2_gpu2b.log" 2>&1; then
  {
    echo "=================================================="
    echo "[END] F06"
    echo "TIME=$(date '+%F %T')"
    echo "=================================================="
  } >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/F06_gpu2_gpu2b.log" 2>&1
  rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F06.running"
  touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F06.ok"
else
  rc=$?
  {
    echo "=================================================="
    echo "[FAIL] F06"
    echo "TIME=$(date '+%F %T')"
    echo "EXIT_CODE=${rc}"
    echo "=================================================="
  } >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/F06_gpu2_gpu2b.log" 2>&1
  rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F06.running"
  touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F06.fail"
  exit "${rc}"
fi

rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F14.ok" "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F14.fail"
touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F14.running"
{
  echo "=================================================="
  echo "[START] F14"
  echo "GPU=2"
  echo "WORKER=gpu2b"
  echo "TIME=$(date '+%F %T')"
  echo "LOG=/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/F14_gpu2_gpu2b.log"
  echo "CMD=python -u main.py --dataset Cifar10 --model resnet --resnet_pretrained 0 --algorithm FedRT --learning_rate 0.1 --lr_head 0.1 --plocal_epochs 1 --local_epochs 20 --num_global_iters 150 --numusers 10 --batch_size 64 --attack_start 30 --attack_method attackall --poisoning_per_batch 5 --defense none --per_epoch 1 --malclient 10 --times 1 --lr_head 0.02 --rt_beta 0.10 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10 "
  echo "=================================================="
} >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/F14_gpu2_gpu2b.log" 2>&1

if python -u main.py --dataset Cifar10 --model resnet --resnet_pretrained 0 --algorithm FedRT --learning_rate 0.1 --lr_head 0.1 --plocal_epochs 1 --local_epochs 20 --num_global_iters 150 --numusers 10 --batch_size 64 --attack_start 30 --attack_method attackall --poisoning_per_batch 5 --defense none --per_epoch 1 --malclient 10 --times 1 --lr_head 0.02 --rt_beta 0.10 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10  >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/F14_gpu2_gpu2b.log" 2>&1; then
  {
    echo "=================================================="
    echo "[END] F14"
    echo "TIME=$(date '+%F %T')"
    echo "=================================================="
  } >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/F14_gpu2_gpu2b.log" 2>&1
  rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F14.running"
  touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F14.ok"
else
  rc=$?
  {
    echo "=================================================="
    echo "[FAIL] F14"
    echo "TIME=$(date '+%F %T')"
    echo "EXIT_CODE=${rc}"
    echo "=================================================="
  } >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/F14_gpu2_gpu2b.log" 2>&1
  rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F14.running"
  touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024141/status/F14.fail"
  exit "${rc}"
fi

echo "[WORKER_END] gpu2b GPU=2 TIME=$(date '+%F %T')"
