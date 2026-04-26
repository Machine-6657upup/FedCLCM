#!/usr/bin/env bash
set -euo pipefail
cd "/home/huangtu/PFL_clean_workspace/root_static/src"
PY="/home/huangtu/miniconda3/envs/torch/bin/python"

"${PY}" -u main.py   -dev cuda -did 3   -data Cifar10_dir0.5_bdoor0.2_nclient_100_badnet_adv10   -m ResNet18 -algo FedCLCM   -ncl 10 -nc 100 -jr 0.1 -lbs 64   -lr 0.1 -ls 1   -gr 600 -eg 1   -go s3_clcm_badnet   --num_adv_clients 10   > "/home/huangtu/PFL_clean_workspace/root_static/runs/20260417_184038_stage3_static_paper_common/train_logs/s3_clcm_badnet.log" 2>&1

"${PY}" -u main.py   -dev cuda -did 3   -data Cifar10_dir0.5_bdoor0.2_nclient_100_blend_adv10   -m ResNet18 -algo FedCLCM   -ncl 10 -nc 100 -jr 0.1 -lbs 64   -lr 0.1 -ls 1   -gr 600 -eg 1   -go s3_clcm_blend   --num_adv_clients 10   > "/home/huangtu/PFL_clean_workspace/root_static/runs/20260417_184038_stage3_static_paper_common/train_logs/s3_clcm_blend.log" 2>&1

"${PY}" -u main.py   -dev cuda -did 3   -data Cifar10_dir0.5_bdoor0.2_nclient_100_sig_adv10   -m ResNet18 -algo FedCLCM   -ncl 10 -nc 100 -jr 0.1 -lbs 64   -lr 0.1 -ls 1   -gr 600 -eg 1   -go s3_clcm_sig   --num_adv_clients 10   > "/home/huangtu/PFL_clean_workspace/root_static/runs/20260417_184038_stage3_static_paper_common/train_logs/s3_clcm_sig.log" 2>&1
