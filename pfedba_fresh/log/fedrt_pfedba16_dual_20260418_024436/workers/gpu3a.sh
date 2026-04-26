#!/usr/bin/env bash
set -euo pipefail
cd "/home/huangtu/PFL_Backdoor_Defense/PFedBA"
export CUDA_VISIBLE_DEVICES="3"
export PYTHONUNBUFFERED=1
echo "[WORKER_START] gpu3a GPU=3 TIME=$(date '+%F %T')"
rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F07.ok" "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F07.fail"
touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F07.running"
{
  echo "=================================================="
  echo "[START] F07"
  echo "GPU=3"
  echo "WORKER=gpu3a"
  echo "TIME=$(date '+%F %T')"
  echo "LOG=/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/F07_gpu3_gpu3a.log"
  echo "CMD=/home/huangtu/miniconda3/envs/torch/bin/python -u /home/huangtu/PFL_Backdoor_Defense/PFedBA/main.py --dataset Cifar10 --model resnet --resnet_pretrained 0 --algorithm FedRT --learning_rate 0.1 --lr_head 0.1 --plocal_epochs 1 --local_epochs 20 --num_global_iters 150 --numusers 10 --batch_size 64 --attack_start 30 --attack_method attackall --poisoning_per_batch 5 --defense none --per_epoch 1 --malclient 10 --times 1 --rt_beta 0.10 --adv_eps 0.12 --adv_num_iter 5 --aug_strength 0.10 "
  echo "=================================================="
} >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/F07_gpu3_gpu3a.log" 2>&1

if /home/huangtu/miniconda3/envs/torch/bin/python -u /home/huangtu/PFL_Backdoor_Defense/PFedBA/main.py --dataset Cifar10 --model resnet --resnet_pretrained 0 --algorithm FedRT --learning_rate 0.1 --lr_head 0.1 --plocal_epochs 1 --local_epochs 20 --num_global_iters 150 --numusers 10 --batch_size 64 --attack_start 30 --attack_method attackall --poisoning_per_batch 5 --defense none --per_epoch 1 --malclient 10 --times 1 --rt_beta 0.10 --adv_eps 0.12 --adv_num_iter 5 --aug_strength 0.10  >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/F07_gpu3_gpu3a.log" 2>&1; then
  {
    echo "=================================================="
    echo "[END] F07"
    echo "TIME=$(date '+%F %T')"
    echo "=================================================="
  } >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/F07_gpu3_gpu3a.log" 2>&1
  rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F07.running"
  touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F07.ok"
else
  rc=$?
  {
    echo "=================================================="
    echo "[FAIL] F07"
    echo "TIME=$(date '+%F %T')"
    echo "EXIT_CODE=${rc}"
    echo "=================================================="
  } >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/F07_gpu3_gpu3a.log" 2>&1
  rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F07.running"
  touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F07.fail"
  exit "${rc}"
fi

rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F15.ok" "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F15.fail"
touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F15.running"
{
  echo "=================================================="
  echo "[START] F15"
  echo "GPU=3"
  echo "WORKER=gpu3a"
  echo "TIME=$(date '+%F %T')"
  echo "LOG=/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/F15_gpu3_gpu3a.log"
  echo "CMD=/home/huangtu/miniconda3/envs/torch/bin/python -u /home/huangtu/PFL_Backdoor_Defense/PFedBA/main.py --dataset Cifar10 --model resnet --resnet_pretrained 0 --algorithm FedRT --learning_rate 0.1 --lr_head 0.1 --plocal_epochs 1 --local_epochs 20 --num_global_iters 150 --numusers 10 --batch_size 64 --attack_start 30 --attack_method attackall --poisoning_per_batch 5 --defense none --per_epoch 1 --malclient 10 --times 1 --learning_rate 0.05 --lr_head 0.05 --rt_beta 0.10 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10 "
  echo "=================================================="
} >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/F15_gpu3_gpu3a.log" 2>&1

if /home/huangtu/miniconda3/envs/torch/bin/python -u /home/huangtu/PFL_Backdoor_Defense/PFedBA/main.py --dataset Cifar10 --model resnet --resnet_pretrained 0 --algorithm FedRT --learning_rate 0.1 --lr_head 0.1 --plocal_epochs 1 --local_epochs 20 --num_global_iters 150 --numusers 10 --batch_size 64 --attack_start 30 --attack_method attackall --poisoning_per_batch 5 --defense none --per_epoch 1 --malclient 10 --times 1 --learning_rate 0.05 --lr_head 0.05 --rt_beta 0.10 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10  >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/F15_gpu3_gpu3a.log" 2>&1; then
  {
    echo "=================================================="
    echo "[END] F15"
    echo "TIME=$(date '+%F %T')"
    echo "=================================================="
  } >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/F15_gpu3_gpu3a.log" 2>&1
  rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F15.running"
  touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F15.ok"
else
  rc=$?
  {
    echo "=================================================="
    echo "[FAIL] F15"
    echo "TIME=$(date '+%F %T')"
    echo "EXIT_CODE=${rc}"
    echo "=================================================="
  } >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/F15_gpu3_gpu3a.log" 2>&1
  rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F15.running"
  touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F15.fail"
  exit "${rc}"
fi

echo "[WORKER_END] gpu3a GPU=3 TIME=$(date '+%F %T')"
