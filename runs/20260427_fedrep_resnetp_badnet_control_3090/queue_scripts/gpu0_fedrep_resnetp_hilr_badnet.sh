#!/usr/bin/env bash
set -euo pipefail
PROJECT=/home/huangtu/FedCLCM_purify
SRC=$PROJECT/src
PY=/home/huangtu/miniconda3/envs/torch/bin/python
RUN=$PROJECT/runs/20260427_fedrep_resnetp_badnet_control_3090
LOG=$RUN/train_logs/FEDREP_RP_HILR_BADNET_ADV10.log
echo [START] $(date '+%F %T') FedRep ResNetP high-lr badnet control | tee -a $RUN/launcher_trace.txt
echo dataset=Cifar10_dir0.5_bdoor0.2_nclient_100_badnet_adv10 model=ResNetP algo=FedRep nc=100 jr=0.1 adv=10 lr=0.1 lr_head=0.1 ls=1 pls=1 gr=600 eg=10 gpu=0 | tee -a $RUN/launcher_trace.txt
cd $SRC
$PY -u main.py \
  -dev cuda -did 0 \
  -data Cifar10_dir0.5_bdoor0.2_nclient_100_badnet_adv10 \
  -m ResNetP -algo FedRep \
  -ncl 10 -nc 100 -jr 0.1 -lbs 64 \
  -lr 0.1 -lr_head 0.1 -ls 1 -pls 1 \
  -gr 600 -eg 10 \
  -go FEDREP_RP_HILR_BADNET_ADV10 \
  --num_adv_clients 10 \
  > $LOG 2>&1
