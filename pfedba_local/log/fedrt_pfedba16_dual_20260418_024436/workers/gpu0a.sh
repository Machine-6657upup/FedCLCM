#!/usr/bin/env bash
set -euo pipefail
cd "/home/huangtu/PFL_Backdoor_Defense/PFedBA"
export CUDA_VISIBLE_DEVICES="0"
export PYTHONUNBUFFERED=1
echo "[WORKER_START] gpu0a GPU=0 TIME=$(date '+%F %T')"
rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F01.ok" "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F01.fail"
touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F01.running"
{
  echo "=================================================="
  echo "[START] F01"
  echo "GPU=0"
  echo "WORKER=gpu0a"
  echo "TIME=$(date '+%F %T')"
  echo "LOG=/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/F01_gpu0_gpu0a.log"
  echo "CMD=/home/huangtu/miniconda3/envs/torch/bin/python -u /home/huangtu/PFL_Backdoor_Defense/PFedBA/main.py --dataset Cifar10 --model resnet --resnet_pretrained 0 --algorithm FedRT --learning_rate 0.1 --lr_head 0.1 --plocal_epochs 1 --local_epochs 20 --num_global_iters 150 --numusers 10 --batch_size 64 --attack_start 30 --attack_method attackall --poisoning_per_batch 5 --defense none --per_epoch 1 --malclient 10 --times 1 --rt_beta 0.10 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10 "
  echo "=================================================="
} >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/F01_gpu0_gpu0a.log" 2>&1

if /home/huangtu/miniconda3/envs/torch/bin/python -u /home/huangtu/PFL_Backdoor_Defense/PFedBA/main.py --dataset Cifar10 --model resnet --resnet_pretrained 0 --algorithm FedRT --learning_rate 0.1 --lr_head 0.1 --plocal_epochs 1 --local_epochs 20 --num_global_iters 150 --numusers 10 --batch_size 64 --attack_start 30 --attack_method attackall --poisoning_per_batch 5 --defense none --per_epoch 1 --malclient 10 --times 1 --rt_beta 0.10 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10  >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/F01_gpu0_gpu0a.log" 2>&1; then
  {
    echo "=================================================="
    echo "[END] F01"
    echo "TIME=$(date '+%F %T')"
    echo "=================================================="
  } >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/F01_gpu0_gpu0a.log" 2>&1
  rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F01.running"
  touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F01.ok"
else
  rc=$?
  {
    echo "=================================================="
    echo "[FAIL] F01"
    echo "TIME=$(date '+%F %T')"
    echo "EXIT_CODE=${rc}"
    echo "=================================================="
  } >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/F01_gpu0_gpu0a.log" 2>&1
  rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F01.running"
  touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F01.fail"
  exit "${rc}"
fi

rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F09.ok" "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F09.fail"
touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F09.running"
{
  echo "=================================================="
  echo "[START] F09"
  echo "GPU=0"
  echo "WORKER=gpu0a"
  echo "TIME=$(date '+%F %T')"
  echo "LOG=/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/F09_gpu0_gpu0a.log"
  echo "CMD=/home/huangtu/miniconda3/envs/torch/bin/python -u /home/huangtu/PFL_Backdoor_Defense/PFedBA/main.py --dataset Cifar10 --model resnet --resnet_pretrained 0 --algorithm FedRT --learning_rate 0.1 --lr_head 0.1 --plocal_epochs 1 --local_epochs 20 --num_global_iters 150 --numusers 10 --batch_size 64 --attack_start 30 --attack_method attackall --poisoning_per_batch 5 --defense none --per_epoch 1 --malclient 10 --times 1 --rt_beta 0.05 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10 "
  echo "=================================================="
} >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/F09_gpu0_gpu0a.log" 2>&1

if /home/huangtu/miniconda3/envs/torch/bin/python -u /home/huangtu/PFL_Backdoor_Defense/PFedBA/main.py --dataset Cifar10 --model resnet --resnet_pretrained 0 --algorithm FedRT --learning_rate 0.1 --lr_head 0.1 --plocal_epochs 1 --local_epochs 20 --num_global_iters 150 --numusers 10 --batch_size 64 --attack_start 30 --attack_method attackall --poisoning_per_batch 5 --defense none --per_epoch 1 --malclient 10 --times 1 --rt_beta 0.05 --adv_eps 0.10 --adv_num_iter 5 --aug_strength 0.10  >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/F09_gpu0_gpu0a.log" 2>&1; then
  {
    echo "=================================================="
    echo "[END] F09"
    echo "TIME=$(date '+%F %T')"
    echo "=================================================="
  } >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/F09_gpu0_gpu0a.log" 2>&1
  rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F09.running"
  touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F09.ok"
else
  rc=$?
  {
    echo "=================================================="
    echo "[FAIL] F09"
    echo "TIME=$(date '+%F %T')"
    echo "EXIT_CODE=${rc}"
    echo "=================================================="
  } >> "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/F09_gpu0_gpu0a.log" 2>&1
  rm -f "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F09.running"
  touch "/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_pfedba16_dual_20260418_024436/status/F09.fail"
  exit "${rc}"
fi

echo "[WORKER_END] gpu0a GPU=0 TIME=$(date '+%F %T')"
