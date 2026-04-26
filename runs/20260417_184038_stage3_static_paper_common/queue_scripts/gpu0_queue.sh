#!/usr/bin/env bash
set -euo pipefail
cd "/home/huangtu/PFL_clean_workspace/root_static/src"
PY="/home/huangtu/miniconda3/envs/torch/bin/python"

"${PY}" -u main.py   -dev cuda -did 0   -data Cifar10_dir0.5_bdoor0.2_nclient_100_badnet_adv10   -m ResNet18 -algo FedAvg   -ncl 10 -nc 100 -jr 0.1 -lbs 64   -lr 0.1 -ls 1   -gr 600 -eg 1   -go s3_avg_badnet   --num_adv_clients 10   > "/home/huangtu/PFL_clean_workspace/root_static/runs/20260417_184038_stage3_static_paper_common/train_logs/s3_avg_badnet.log" 2>&1

"${PY}" -u main.py   -dev cuda -did 0   -data Cifar10_dir0.5_bdoor0.2_nclient_100_badnet_adv10   -m ResNet18 -algo FedRep   -ncl 10 -nc 100 -jr 0.1 -lbs 64   -lr 0.1 -lr_head 0.1 -ls 1 -pls 1   -gr 600 -eg 1   -go s3_rep_badnet   --num_adv_clients 10   > "/home/huangtu/PFL_clean_workspace/root_static/runs/20260417_184038_stage3_static_paper_common/train_logs/s3_rep_badnet.log" 2>&1
