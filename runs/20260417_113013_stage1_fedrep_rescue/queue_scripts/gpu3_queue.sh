#!/usr/bin/env bash
set -euo pipefail
cd "/home/huangtu/PFL_clean_workspace/root_static/src"
PY="/home/huangtu/miniconda3/envs/torch/bin/python"

"${PY}" -u main.py   -dev cuda -did 3   -data Cifar10_dir0.5_bdoor0.2_nclient_40_badnet_adv5   -m ResNet18 -algo FedRep   -ncl 10 -nc 40 -jr 1.0 -lbs 64   -lr 0.01 -lr_head 0.01 -ls 1 -pls 1   -gr 800 -eg 1   -go s1_r18_j1_lr0p01_h0p01   --num_adv_clients 5   > "/home/huangtu/PFL_clean_workspace/root_static/runs/20260417_113013_stage1_fedrep_rescue/train_logs/s1_r18_j1_lr0p01_h0p01.log" 2>&1

"${PY}" -u main.py   -dev cuda -did 3   -data Cifar10_dir0.5_bdoor0.2_nclient_40_badnet_adv5   -m ResNetP -algo FedRep   -ncl 10 -nc 40 -jr 0.1 -lbs 64   -lr 0.01 -lr_head 0.03 -ls 1 -pls 1   -gr 800 -eg 1   -go s1_rp_j0p1_lr0p01_h0p03   --num_adv_clients 5   > "/home/huangtu/PFL_clean_workspace/root_static/runs/20260417_113013_stage1_fedrep_rescue/train_logs/s1_rp_j0p1_lr0p01_h0p03.log" 2>&1
