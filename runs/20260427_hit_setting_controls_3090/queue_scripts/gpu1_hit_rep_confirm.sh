#!/usr/bin/env bash
set -euo pipefail
PROJECT=/home/huangtu/FedCLCM_purify
SRC=$PROJECT/src
PY=/home/huangtu/miniconda3/envs/torch/bin/python
RUN=$PROJECT/runs/20260427_hit_setting_controls_3090
LOG=$RUN/train_logs/HIT_REP_R18_CONFIRM.log
echo [START] $(date '+%F %T') HIT_REP_R18_CONFIRM gpu=1 | tee -a $RUN/launcher_trace.txt
cd $SRC
$PY -u main.py -dev cuda -did 1 -data Cifar10_dir0.5_bdoor0.2_nclient_100_badnet_adv10 -m ResNet18 -algo FedRep -ncl 10 -nc 100 -jr 0.1 -lbs 64 -lr 0.1 -lr_head 0.1 -ls 1 -pls 1 -gr 600 -eg 10 -go HIT_REP_R18_CONFIRM --num_adv_clients 10 > $LOG 2>&1
