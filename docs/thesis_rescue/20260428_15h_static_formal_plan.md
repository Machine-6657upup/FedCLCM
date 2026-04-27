# 2026-04-28 15h Static Formal Plan

## Fixed Thesis Setting

- Dataset: `Cifar10_dir0.5_bdoor0.2_nclient_100_{badnet,blend,sig}_adv10`.
- Clients: `num_clients=100`, `num_adv_clients=10`, `join_ratio=0.1`.
- Model: non-pretrained `ResNet18`.
- Optimizer-critical params: `lr=0.1`, `lr_head=0.1`, `local_epochs=1`, `plocal_epochs=1`, `batch_size=64`.
- FedCLCM default: `rt_beta=0.2`, `lambda_cl=0.2`, `aug_strength=0.1`, `mask_tau=6.0`, `mask_alpha=0.3`, `adv_eps=0`, `adv_num_iter=0`.
- New formal runs use `global_rounds=800`, `eval_gap=10`.
- Existing 600-round logs can be reused only when parameters match and round 500-600 has clearly converged.

## Reused Logs

- `thesis_formal_logs/final_static_main_20260424/static/S*.log`: FedAvg/FedRep static attack calibration logs. These are reused instead of rerun because the key thesis point is that static attacks can break FedAvg/FedRep under the correct common hyperparams.
- `runs/20260427_hit_setting_t6a03_baseline_3090/train_logs/HIT_CLCM_T6A03_BASE.log`: current FedCLCM BadNet setting, 600 rounds, stable and usable.
- `runs/20260427_pfedba_clcmpur_priority_4090/train_logs/PF01/PF03`: PFedBA reference logs only, not the static main table.

## New 3090 Queue

- GPU0 runs missing FedCLCM static main results and BadNet ablations/sensitivity.
- GPU1 runs coordinate robust baselines: `FedMedian`, `FedTrimmed`.
- GPU2 runs Krum-family robust baselines: `FedMK`, `FedBulyan`.
- GPU3 runs PFL baselines: `FedProto`, `FedPD`.

## New 4090 Queue

- Runs FedCLCM ablations on Blend/SIG.
- Runs BadNet local epoch sensitivity with `local_epochs=2` and `local_epochs=5`.
- Runs stronger mask sensitivity on Blend/SIG with `tau=5.0`, `alpha=0.2`.

## Baseline Implementation Notes

- `FedMedian` is coordinate-wise median over uploaded model parameters.
- `FedTrimmed` was revised to use the current round selected malicious count `f`, matching the PFedBA reference style that computes `m` from selected malicious users before trimming.
- `FedMK` was revised from fixed `m=8,k=7` to a Multi-Krum-style score using the current round selected malicious count `f`.
- `FedBulyan` is marked Bulyan-style. It uses Krum-style candidate selection plus coordinate trimmed mean when `n >= 4f+3`; otherwise it falls back to trimmed mean because exact Bulyan is not well-defined for that round.
- `FedProto` follows the local prototype aggregation code that explicitly references the official FedProto implementation by Yu et al.
- `FedPD` is a local prototype-defense variant, not treated as a canonical reproduction of every FedPD paper.

## Why This Plan

- The main method remains FedCLCM, not CLCMPur.
- The experiment table needs more baselines without rerunning already converged FedAvg/FedRep logs.
- Robust aggregation baselines are included because PFedBA and related defenses often compare against Median, Trimmed Mean, Multi-Krum and Bulyan.
- FedProto/FedPD are included because the thesis is about personalized FL, so only robust aggregation baselines are not enough.
