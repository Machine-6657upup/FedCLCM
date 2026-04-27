#!/usr/bin/env bash
set -euo pipefail
PROJECT=/home/fch/FedCLCM_purify
SRC=$PROJECT/src
PY=/home/fch/miniconda3/envs/fedclcm-static/bin/python
RUN=$PROJECT/runs/20260427_hit_setting_clcmpur_grid_4090
LOG=$RUN/train_logs/HIT_PUR_B0P1_S50.log
echo [START] $(date '+%F %T') HIT_PUR_B0P1_S50 gpu=0 | tee -a $RUN/launcher_trace.txt
cd $SRC
$PY -u main.py -dev cuda -did 0 -data Cifar10_dir0.5_bdoor0.2_nclient_100_badnet_adv10 -m ResNet18 -algo FedCLCMPurify -ncl 10 -nc 100 -jr 0.1 -lbs 64 -lr 0.1 -lr_head 0.1 -ls 1 -pls 1 -gr 600 -eg 10 -go HIT_PUR_B0P1_S50 --num_adv_clients 10 --rt_beta 0.2 --lambda_cl 0.2 --aug_strength 0.1 --mask_tau 12.0 --mask_alpha 0.7 --enable_channel_mask true --adv_eps 0.0 --adv_num_iter 0 --purify_beta 0.1 --purify_feature_beta 0.0 --purify_logit_beta 0.0 --purify_start_round 50 --purify_layers layer4 --purify_teacher_momentum 0.90 --purify_teacher_cpu_half true > $LOG 2>&1
