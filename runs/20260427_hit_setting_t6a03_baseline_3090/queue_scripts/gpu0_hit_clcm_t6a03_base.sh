#!/usr/bin/env bash
set -euo pipefail
PROJECT=/home/huangtu/FedCLCM_purify
SRC=$PROJECT/src
PY=/home/huangtu/miniconda3/envs/torch/bin/python
RUN=$PROJECT/runs/20260427_hit_setting_t6a03_baseline_3090
LOG=$RUN/train_logs/HIT_CLCM_T6A03_BASE.log
echo [START] $(date '+%F %T') HIT_CLCM_T6A03_BASE gpu=0 | tee -a $RUN/launcher_trace.txt
cd $SRC
$PY -u main.py -dev cuda -did 0 -data Cifar10_dir0.5_bdoor0.2_nclient_100_badnet_adv10 -m ResNet18 -algo FedCLCM -ncl 10 -nc 100 -jr 0.1 -lbs 64 -lr 0.1 -lr_head 0.1 -ls 1 -pls 1 -gr 600 -eg 10 -go HIT_CLCM_T6A03_BASE --num_adv_clients 10 --rt_beta 0.2 --lambda_cl 0.2 --aug_strength 0.1 --mask_tau 6.0 --mask_alpha 0.3 --enable_channel_mask true --adv_eps 0.0 --adv_num_iter 0 > $LOG 2>&1
