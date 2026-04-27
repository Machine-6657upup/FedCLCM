# 2026-04-28 Master Experiment Matrix

This matrix is for the thesis-scale FedCLCM experiments. New overnight tasks do not rerun FedAvg, FedRep, FedRT or FedRTSD.

## Thesis Questions Covered

- Main effectiveness: FedCLCM against BadNet, Blend and SIG with 800 rounds.
- Defense comparison: Median, Trimmed Mean, Multi-Krum, Bulyan-style aggregation, FedProto and FedPD.
- Component ablation: remove contrastive loss, remove channel mask, remove both.
- Data heterogeneity: Dirichlet alpha 0.1, 0.5 and 1.0.
- Attack strength: backdoor rate 0.1, 0.2 and 0.3.
- Malicious ratio: 5, 10 and 20 adversarial clients out of 100.
- System setting sensitivity: join ratio 0.05, 0.1 and 0.2.
- Training sensitivity: local epochs 1, 2, 5, 10 and 20.
- Hyperparameter sensitivity: `mask_tau`, `mask_alpha` and `lambda_cl`.
- Dynamic attack reference: PFedBA 1000-round FedCLCM logs from 2026-04-27 are copied as reference-only logs.

## Why This Is Thesis-Scale

The formal story is no longer just "three static attacks plus PFedBA". The matrix supports separate thesis sections for effectiveness, comparison with existing defenses, ablation, sensitivity, robustness under stronger settings, and convergence/cost visualization.
