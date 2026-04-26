# Stage3 Static Experiments

## Why Stage3 Exists
Stage3 is the static-attack calibration and comparison stage under one unified common regime.

Its job is not to produce the final thesis chapter by itself. Its job is to restore trust in the static experiment line.

## Common Regime
Stage3 uses:
- `num_clients = 100`
- `join_ratio = 0.1`
- `local_learning_rate = 0.1`
- `local_epochs = 1`
- `plocal_epochs = 1`
- `model = ResNet18`
- `num_adv_clients = 10`
- `global_rounds = 600`
- `batch_size = 64`
- dataset alpha `dirichlet = 0.5`

Dataset naming pattern:
- `Cifar10_dir0.5_bdoor0.2_nclient_100_*`

## Why This Regime Was Chosen
After comparing:
- old project logs
- recent failed runs
- paper conventions
- user recollection

the team converged on the belief that the following combination is broadly correct and credible:
- `num_clients = 100`
- `join_ratio = 0.1`
- `lr = 0.1`

This was specifically treated as more trustworthy than recent runs that drifted toward:
- `join_ratio = 1`
- `lr = 0.01`

## Script And Run Directory
Stage3 launcher:
- `/home/huangtu/PFL_clean_workspace/root_static/scripts/run_stage3_static_paper_common_4x3090.sh`

Run directory:
- `/home/huangtu/PFL_clean_workspace/root_static/runs/20260417_184038_stage3_static_paper_common`

Parsed metric artifacts:
- `/home/huangtu/PFL_clean_workspace/root_static/runs/20260417_184038_stage3_static_paper_common/metrics_summary.csv`
- `/home/huangtu/PFL_clean_workspace/root_static/runs/20260417_184038_stage3_static_paper_common/metrics_summary.json`
- `/home/huangtu/PFL_clean_workspace/root_static/runs/20260417_184038_stage3_static_paper_common/curves/`

Local downloaded logs:
- `C:\Users\12709\remote_edit\root_static\downloads\20260417_stage3_logs`

## Task Layout
GPU0 queue:
- `s3_avg_badnet`
- `s3_rep_badnet`

GPU1 queue:
- `s3_avg_blend`
- `s3_rep_blend`

GPU2 queue:
- `s3_avg_sig`
- `s3_rep_sig`

GPU3 queue:
- `s3_clcm_badnet`
- `s3_clcm_blend`
- `s3_clcm_sig`

## Attack-Generation Parameters Confirmed In This Stage
Blend:
- `blend_alpha = 0.2`

SIG:
- `sig_delta = 30/255`
- `sig_f = 6`
- `sig_label_mode = dirty`

## Current Remote Accounting Update

Accounting status checked directly from the remote logs on `2026-04-18`.

Completion summary:
- fully finished to round `600` with `All done!`:
  - `s3_avg_badnet`
  - `s3_avg_blend`
  - `s3_avg_sig`
  - `s3_rep_badnet`
  - `s3_rep_blend`
  - `s3_rep_sig`
  - `s3_clcm_badnet`
  - `s3_clcm_blend`
- not fully finished:
  - `s3_clcm_sig`
  - the log reaches round header `65`
  - the last complete metric block is round `64`

## Confirmed Results From Remote Logs

### FedAvg
`s3_avg_badnet.log`
- final acc: `0.5130`
- final ASR: `0.9345`
- best acc: `0.5130`
- best round: `600`

`s3_avg_blend.log`
- final acc: `0.5252`
- final ASR: `0.7112`
- best acc: `0.5267`
- best round: `598`

`s3_avg_sig.log`
- final acc: `0.4968`
- final ASR: `0.9934`
- best acc: `0.5188`
- best round: `592`

Interpretation:
- corrected BadNet is strong enough
- corrected SIG is now very strong
- Blend is clearly stronger than the weak earlier state

### FedRep
`s3_rep_badnet.log`
- final acc: `0.7870`
- final ASR: `0.5755`
- best acc: `0.7887`
- best round: `570`

`s3_rep_blend.log`
- final acc: `0.7921`
- final ASR: `0.1794`
- best acc: `0.7954`
- best round: `498`

`s3_rep_sig.log`
- final acc: `0.7906`
- final ASR: `0.3393`
- best acc: `0.7948`
- best round: `523`

Interpretation:
- FedRep is no longer stuck in the low-0.7x region
- static attacks are not "too weak to touch FedRep"
- BadNet is clearly meaningful
- SIG is meaningful
- Blend is weaker than BadNet/SIG, but no longer trivial

This is enough to rescue the thesis static baseline.

### FedCLCM
`s3_clcm_badnet.log`
- final acc: `0.4033`
- final ASR: `0.0462`
- best acc: `0.5091`
- best round: `41`

`s3_clcm_blend.log`
- final acc: `0.4037`
- final ASR: `0.0650`
- best acc: `0.4632`
- best round: `34`

`s3_clcm_sig.log`
- partial only
- best acc so far: `0.4720`
- best round so far: `38`
- last complete round: `64`
- last complete acc: `0.3833`
- last complete ASR: `0.0336`

Interpretation:
- FedCLCM currently has a severe collapse/interpretation problem
- low ASR is coming together with heavily damaged clean performance
- this cannot be treated as a clean defense success

## Convergence Versus Collapse

FedAvg:
- all three runs converge late rather than peaking early
- best rounds are `592`, `598`, and `600`
- final accuracy stays close to the best accuracy
- ASR remains high, so these are stable attack-success runs, not accidental failures

FedRep:
- all three runs also converge late
- best rounds are `498`, `523`, and `570`
- final accuracy is within about `0.002` to `0.004` of best accuracy
- final clean accuracy stays around `0.79`
- ASR is clearly nontrivial under all three attacks

FedCLCM:
- `badnet` and `blend` peak very early at rounds `41` and `34`
- both then spend the rest of training far below their early clean-accuracy peak
- `sig` shows the same pattern so far: best at round `38`, then clean accuracy falls into the `0.38` to `0.44` region by round `64`
- therefore the current FedCLCM static story is collapse with low ASR, not a clean defense victory

## What Stage3 Already Proved
Stage3 already proved the following high-value thesis facts:
1. static attacks are not inherently too weak
2. FedRep is not untouchable
3. a reasonable common regime can recover a credible FedRep clean-accuracy baseline
4. FedCLCM currently does not obviously dominate FedRep in a clean, convincing way

## What Is Still Missing
Only one stage3 item is still incomplete:
- `s3_clcm_sig`

Current practical interpretation:
- the static comparison table is already usable
- the missing item no longer blocks the main thesis claim
- but if a fully completed FedCLCM-SIG point is later needed, only that single run still needs completion or rerun

## Thesis-Level Interpretation
The project no longer needs to say:
"we must reproduce the exact old 88+ result or the thesis is dead."

It can now say:
"under a clean unified setup, FedRep attains solid clean accuracy and static backdoors remain effective enough to provide a meaningful comparison basis."
