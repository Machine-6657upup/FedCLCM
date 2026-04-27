# FedCLCM-Purify Design Check - 2026-04-27

## Purpose

The short-term thesis goal is not to invent many new variants. It is to find one defensible FedCLCM upgrade that keeps clean ACC close to the current/old FedCLCM level while reducing static ASR, especially Badnet.

The current formal static table proves static attacks are no longer too weak because FedRep is attacked, but FedCLCM itself is still not satisfactory:

- Badnet FedCLCM: `ACC=0.757`, `ASR=0.522`
- Blend FedCLCM: `ACC=0.752`, `ASR=0.160`
- SIG FedCLCM: `ACC=0.749`, `ASR=0.311`

The 12h sweep improved Badnet at best to `ACC=0.7897`, `ASR=0.2785`, but this is still too high for a main defense claim.

## Evidence From Old CLCM Logs

The old ablation and sensitivity logs are now locally archived under `remote_old_logs/thesis_log/` and summarized as CSV files under `remote_old_logs/`.

Key old CIFAR-10/ResNetP facts:

- `C4_no_mask`: `ACC=0.7780`, `ASR=0.0514`
- `C1_full_lite`: `ACC=0.7777`, `ASR=0.0609`
- `C3_no_aug`: `ACC=0.7755`, `ASR=0.0730`
- `lambda_0p20`: `ACC=0.7867`, `ASR=0.0500`
- `trim_0p05`: `ACC=0.7804`, `ASR=0.0595`
- `hp_lrh_0p005`: `ACC=0.7889`, `ASR=0.0410`
- `hp_ls_2`: `ACC=0.7981`, `ASR=0.0467`
- `hp_jr_0p25`: `ACC=0.8008`, `ASR=0.0493`

Interpretation:

- The reliable old ingredients are `ResNetP`, `lr=0.003`, `lr_head=0.005/0.01`, `lambda_cl=0.2`, `aug_strength=0.05/0.1`, and `rt_beta=0.05`.
- Mask is not a consistently positive component. The no-mask ablation had lower ASR than full-lite in the old module group.
- More local epochs can reduce ASR but easily hurts ACC. Recent 100-client logs show the same tradeoff: ASR drops for `E=10/20`, but ACC collapses too much.
- Therefore the upgrade should not be "stronger trim + stronger mask". That route has already been tested and is not enough.

## Paper-Grounded Design

BDPFL and PFL-ALP both point to the same useful direction: move the real cleansing step to benign clients instead of relying only on server-side similarity or robust aggregation.

The implemented upgrade is `FedCLCMPurify`:

- Keep the FedCLCM server aggregation unchanged.
- Keep the historical FedCLCM client CE + contrastive learning path unchanged.
- Only benign clients maintain an EMA teacher snapshot of their locally cleaned personalized model.
- After receiving the global/shared base, benign clients add attention alignment from the current base to the local teacher during base training.
- Malicious clients do not get purification.
- The feature/logit distillation hooks exist but default to zero because the first validation target is a minimal, low-risk attention purification.

This matches the usable part of PFL-ALP: a personalized teacher purifies a potentially backdoored representative/global model on clean local data.

## Important Code Check

An initial implementation bug was found before running large experiments:

- If the teacher and student weights are identical, purification loss should be nearly zero.
- With teacher in eval mode and student in train mode, BatchNorm made the loss about `1.017` at `purify_beta=500`.
- This would have optimized BatchNorm-mode mismatch rather than backdoor-related attention mismatch.
- The fix is to run the purification forward for both student base and teacher base with eval-mode BatchNorm, then restore the student's original train mode.
- After the fix, identical weights produce `purification_loss=1.6e-7`, while a perturbed layer4 produces a nonzero loss.

This check is essential because PFL-ALP uses a large NAD weight. Without this fix, `beta=500/1500` would be unsafe and hard to interpret.

## Experiment Script

Script:

`scripts/run_static_fedclcm_purify_rescue.sh`

Profiles:

- `PROFILE=smoke`: two short crash tests.
- `PROFILE=badnet_focus`: paired Badnet validation, including current-best formal recipe, old-success transfer recipe, no-mask check, and historical positive control.
- `PROFILE=static_extension`: only run after Badnet shows a positive tradeoff; extends the chosen recipe to Blend and SIG.
- `PROFILE=all`: badnet focus plus static extension.

The script writes:

- logs to `runs/<run_name>/train_logs/`
- curves to `runs/<run_name>/curves/`
- full parameter manifest to `runs/<run_name>/manifest.tsv`
- parsed summary to `runs/<run_name>/summary.csv`

## Initial Parameter Choices

The purification beta values are not arbitrary:

- PFL-ALP identifies `beta=1500` as the effective NAD value.
- A first direct transfer of `beta=500/1500` into the high-lr `ResNet18, lr=0.1` FedCLCM setting was not stable.
- On 2026-04-27, the first Badnet focus run was stopped early because `BN100_B04_PUR_L4_B500/B1500` had `ACC≈0.295` and train loss up to `8.6e7/4.9e9` by round 30.
- Therefore high-lr ResNet18 purification must use smaller delayed weights first: `beta=10/50`, `purify_start_round=10`.
- The old-success low-lr ResNetP path can still test larger values because `lr=0.003`; the safer grid is `beta=100/500`, `purify_start_round=5`.
- `layer4` is the default purification layer because higher layers carry more backdoor semantics and over-constraining lower/mid layers risks ACC.

The Badnet focus matrix includes:

- `BN100_B04_BASE`: current best formal-style FedCLCM baseline.
- `BN100_B04_PUR_L4_B10_S10/B50_S10`: same setting plus small delayed layer4 purification.
- `BN100_OLD_BASE`: old-success recipe transferred to `nclient=100`.
- `BN100_OLD_PUR_L4_B100_S5/B500_S5`: old-success recipe plus safer delayed layer4 purification.
- `BN100_OLD_PUR_NOMASK_L4_B100_S5`: tests the old no-mask signal under the 100-client scale.
- `BN40_PC_BASE` and `BN40_PC_PUR_L4_B100_S5`: positive controls for reproducing the historical good regime.

Early signal from the stopped first run:

- `BN100_OLD_BASE` had ASR around `0.04-0.05` by round 60 before being stopped with the bad high-beta batch.
- This makes the old-success transfer path the current highest-priority rescue path.

## Decision Rule

Continue this path only if at least one Badnet configuration gives a real tradeoff:

- ACC stays near the corresponding baseline, preferably no more than 2-3 points lower.
- ASR drops clearly below the best current sweep, ideally toward `< 0.10`.

If purification only lowers ASR by collapsing ACC, it is not thesis-useful.

If old-success transfer works but formal ResNet18 does not, the thesis should honestly state that FedCLCM is sensitive to representation/optimization regime and use the stable ResNetP configuration for the main static defense table.

If neither transfer nor purification works, stop spending time on FedCLCM static rescue and switch the thesis claim to a narrower one.
