# Thesis Final Experiment Plan

Date: 2026-04-23

This file fixes the final thesis experiment matrix for the PFedBA-on-FedRep line.

## Goal

Use `FedRep` as the personalized baseline under the static backdoor setting that has already been validated to be attackable, then complete:

1. Main comparison methods for the thesis body.
2. Local-epoch sensitivity.
3. Attack-strength sensitivity.
4. `FedCLCM` ablations.
5. Full paper-style appendix reproductions for `PFLALP` and `BDPFL`.

## Common Setting

Shared default arguments:

- `dataset=Cifar10`
- `model=resnet`
- `resnet_pretrained=0`
- `learning_rate=0.1`
- `lr_head=0.1`
- `plocal_epochs=1`
- `total_users=100`
- `numusers=10`
- `batch_size=64`
- `attack_start=30`
- `attack_method=attackall`
- `poisoning_per_batch=1`
- `defense=none`
- `per_epoch=1`
- `malclient=10`
- `times=3`
- `seed=1`

Evaluation focus for the thesis main table:

- `Average Personal Accurancy (k local SGD)`
- `Average Personal ATTACK ALL ASR (k local SGD)`

## Validated Method Hyperparameters

`FedRT`:

- `rt_beta=0.10`
- `adv_eps=0.10`
- `adv_num_iter=5`
- `aug_strength=0.10`

`FedCLCM`:

- `rt_beta=0.20`
- `lambda_cl=0.20`
- `mask_tau=12.0`
- `mask_alpha=0.70`
- `adv_eps=0`
- `adv_num_iter=0`
- `enable_channel_mask=1`
- `cosine_gate=0`

## Final Thesis Matrix

### Group `main`

- `T01` `FedAvg`, `local_epochs=1`, `num_global_iters=1000`
- `T02` `FedAvg + MKrum`, `local_epochs=1`, `num_global_iters=1000`
- `T03` `FedAvg + Trimmed Mean`, `local_epochs=1`, `num_global_iters=1000`
- `T04` `FedRep`, `local_epochs=1`, `num_global_iters=1000`
- `T05` `FedRT`, `local_epochs=1`, `num_global_iters=1000`, plus `FedRT` hyperparameters
- `T06` `FedCLCM`, `local_epochs=1`, `num_global_iters=1000`, plus `FedCLCM` hyperparameters

### Group `epoch`

- `T07` `FedRep`, `local_epochs=10`, `num_global_iters=400`
- `T08` `FedRT`, `local_epochs=10`, `num_global_iters=400`, plus `FedRT` hyperparameters
- `T09` `FedCLCM`, `local_epochs=10`, `num_global_iters=400`, plus `FedCLCM` hyperparameters

### Group `attackers`

- `T10` `FedRep`, `malclient=0`, `local_epochs=1`, `num_global_iters=1000`
- `T11` `FedRT`, `malclient=0`, `local_epochs=1`, `num_global_iters=1000`
- `T12` `FedCLCM`, `malclient=0`, `local_epochs=1`, `num_global_iters=1000`
- `T13` `FedRep`, `malclient=1`, `local_epochs=1`, `num_global_iters=1000`
- `T14` `FedRT`, `malclient=1`, `local_epochs=1`, `num_global_iters=1000`
- `T15` `FedCLCM`, `malclient=1`, `local_epochs=1`, `num_global_iters=1000`
- `T16` `FedRep`, `malclient=50`, `local_epochs=1`, `num_global_iters=1000`
- `T17` `FedRT`, `malclient=50`, `local_epochs=1`, `num_global_iters=1000`
- `T18` `FedCLCM`, `malclient=50`, `local_epochs=1`, `num_global_iters=1000`

### Group `ppb`

- `T19` `FedRep`, `poisoning_per_batch=3`, `local_epochs=1`, `num_global_iters=1000`
- `T20` `FedRT`, `poisoning_per_batch=3`, `local_epochs=1`, `num_global_iters=1000`
- `T21` `FedCLCM`, `poisoning_per_batch=3`, `local_epochs=1`, `num_global_iters=1000`
- `T22` `FedRep`, `poisoning_per_batch=5`, `local_epochs=1`, `num_global_iters=1000`
- `T23` `FedRT`, `poisoning_per_batch=5`, `local_epochs=1`, `num_global_iters=1000`
- `T24` `FedCLCM`, `poisoning_per_batch=5`, `local_epochs=1`, `num_global_iters=1000`

Note:

- `poisoning_per_batch=1` is already covered by the `main` group and reused as the baseline point.

### Group `ablation`

All ablations inherit the validated `FedCLCM` setting except the changed item.

- `T25` `FedCLCM no_trim`: `rt_beta=0.00`
- `T26` `FedCLCM weak_trim`: `rt_beta=0.10`
- `T27` `FedCLCM no_cl`: `lambda_cl=0.00`
- `T28` `FedCLCM no_mask`: `enable_channel_mask=0`

### Group `appendix`

These are not the thesis main-table methods. They are appendix-style full reproductions aligned to the original defense papers as far as the current code line allows.

- `T29` `PFLALP`
  `attack_start=0`
  `selection_strategy=fixed_malicious_mix`
  `fixed_malicious_per_round=3`
  `malclient_id_mode=seeded_pool`
  `malclient=30`
  `local_epochs=1`
  `num_global_iters=100`
  `purify_beta=1500`
  `purify_rounds=1`
  `cluster_max_k=4`
  `mal_local_epoch=6`
  `mal_learning_rate=0.05`
  `times=1`
  `seed=1`

- `T30` `BDPFL`
  `attack_start=0`
  `selection_strategy=fixed_malicious_mix`
  `fixed_malicious_per_round=3`
  `malclient_id_mode=seeded_pool`
  `malclient=3`
  `local_epochs=20`
  `num_global_iters=1000`
  `lr_decay=0.99`
  `lr_decay_step=10`
  `bd_lambda=1`
  `bd_tau=1`
  `bd_gamma=1`
  `bd_use_inter=1`
  `bd_use_em=1`
  `times=1`
  `seed=1`

## Why This Matrix

- The thesis body keeps only methods with clear comparison value on the current PFedBA-on-FedRep setting.
- `FedCLCM` is the strongest current result and must be the main defense line.
- `FedRT` stays as the `AdvPurge` line and remains a required comparison baseline.
- `FedAvg + MKrum/Trimmed Mean` are kept as classical non-personalized baselines.
- Poor FedRep-transfer lite modules such as `FedRepPFLALP cluster_only`, `FedRepPFLALP cluster+purify`, `FedRepBDPFL inter_only`, and `FedRepBDPFL inter+em` are excluded from the thesis main table.

## Script

Main launcher:

- `/home/huangtu/PFL_clean_workspace/root_static/pfedba_local/scripts/run_thesis_final_4gpu.sh`

The launcher supports `screen` or `nohup`, writes a manifest, checks GPU visibility, isolates one queue per GPU, and supports partial rerun by group or tag.

## Launch Examples

Full suite with detached `screen` workers:

```bash
RUN_TS=thesis_final_$(date +%Y%m%d_%H%M%S) \
LAUNCH_MODE=screen \
bash /home/huangtu/PFL_clean_workspace/root_static/pfedba_local/scripts/run_thesis_final_4gpu.sh
```

Only thesis main table plus local-epoch study:

```bash
RUN_TS=thesis_main_epoch_$(date +%Y%m%d_%H%M%S) \
LAUNCH_MODE=screen \
RUN_GROUPS=main,epoch \
bash /home/huangtu/PFL_clean_workspace/root_static/pfedba_local/scripts/run_thesis_final_4gpu.sh
```

Only attacker-count and poisoning-strength sensitivity:

```bash
RUN_TS=thesis_sensitivity_$(date +%Y%m%d_%H%M%S) \
LAUNCH_MODE=screen \
RUN_GROUPS=attackers,ppb \
bash /home/huangtu/PFL_clean_workspace/root_static/pfedba_local/scripts/run_thesis_final_4gpu.sh
```

Only `FedCLCM` ablations:

```bash
RUN_TS=thesis_ablation_$(date +%Y%m%d_%H%M%S) \
LAUNCH_MODE=screen \
RUN_GROUPS=ablation \
bash /home/huangtu/PFL_clean_workspace/root_static/pfedba_local/scripts/run_thesis_final_4gpu.sh
```

Only appendix paper-style full reproductions:

```bash
RUN_TS=thesis_appendix_$(date +%Y%m%d_%H%M%S) \
LAUNCH_MODE=screen \
RUN_GROUPS=appendix \
bash /home/huangtu/PFL_clean_workspace/root_static/pfedba_local/scripts/run_thesis_final_4gpu.sh
```

Use `nohup` instead of `screen`:

```bash
RUN_TS=thesis_final_$(date +%Y%m%d_%H%M%S) \
LAUNCH_MODE=nohup \
bash /home/huangtu/PFL_clean_workspace/root_static/pfedba_local/scripts/run_thesis_final_4gpu.sh
```

## Reproducibility Note

`main.py` now exposes `--seed`, and repeated runs use `seed + i`, so `times=3` is a defensible thesis setting instead of repeating the exact same malicious-client draw every time.
