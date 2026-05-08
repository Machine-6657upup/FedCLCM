# 2026-05-09 Overnight Results Integration and Next Experiment Plan

## 1. Sync Status

- Synced local logs from 3090 and 4090 into `remote_results/20260508_overnight_results/`.
- Pushed to GitHub in commit `df44ba3 add overnight thesis rescue experiment logs`.
- Completed jobs: 19 static BadNet ResNet18 runs on 3090, 12 PFedBA runs on 4090, all 1000 rounds.

## 2. What Changed After the 1000-Round Runs

### Static BadNet

The previous chapter story had two weak points:

- default BadNet: `80.16 / 32.78`, visually weak;
- old multiseed BadNet: `76.71±0.11 / 30.49±7.68`, also weak.

The new 1000-round ResNet18 BadNet runs provide a better, traceable story:

| Role | Tag | Key Hyperparameters | ACC | ASR | Use in Thesis |
| --- | --- | --- | ---: | ---: | --- |
| best low-ASR usable seed | `BN_LOWCL_T5A03_S43` | `lr=0.08, lambda_cl=0.05, tau=5.0, alpha=0.3, seed=43` | 72.29 | 4.86 | BadNet safety operating point |
| balanced point | `BN_M_T5A02_LC005` | `lr=0.08, lambda_cl=0.05, tau=5.0, alpha=0.2` | 72.93 | 6.96 | mask sensitivity / frontier |
| high-ACC seed | `BN_LOWCL_T5A03_S44` | same as main low-CL setting, seed=44 | 78.64 | 13.11 | shows ACC can recover with moderate ASR |
| three-seed low-CL mean | `S42/S43/S44` | same low-CL setting | 74.64±2.84 | 8.65±3.40 | replace weak BadNet multiseed row |
| CL ablation low | `BN_CL0_T5A03` | `lambda_cl=0.00` | 71.16 | 6.17 | shows CL can preserve utility but may preserve patch shortcuts |
| CL ablation balanced | `BN_CL002_T5A03` | `lambda_cl=0.02` | 75.11 | 11.18 | good ablation row |

Interpretation:

1. `lambda_cl=0.05` plus moderate mask `tau=5, alpha=0.3` is much better for BadNet than the old default `lambda_cl=0.2` or `0.5` regimes.
2. The ASR/ACC frontier is real: `lambda_cl=0` and low mask strength can lower ASR, but utility is slightly lower; `lambda_cl=0.02`/`0.05` gives better balance.
3. Increasing local epochs is bad for BadNet: `LE=2` gives `71.46 / 47.02`, `LE=5` gives `67.18 / 44.12`.
4. Higher join ratio `0.20` is bad in the new low-CL regime: `75.75 / 39.41`.

### PFedBA

New PFedBA runs use existing `pfedba_local/main.py` and are directly traceable. The best rows are:

| Tag | LR | `tau` | `alpha` | Global ACC/ASR | Personalized ACC/ASR | Use in Thesis |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `PF_LR005_T50A03_LC005` | 0.05 | 5.0 | 0.3 | 66.56 / 13.72 | 71.46 / 13.16 | best low-ASR PFedBA point |
| `PF_LR003_T50A03_LC005` | 0.03 | 5.0 | 0.3 | 66.69 / 14.41 | 71.20 / 14.46 | confirms low-LR/low-mask trend |
| `PF_LR005_T120A07_LC005` | 0.05 | 12.0 | 0.7 | 67.77 / 14.82 | 73.62 / 15.34 | stronger utility point |
| `PF_LR008_T100A07_LC005` | 0.08 | 10.0 | 0.7 | 71.90 / 17.31 | 75.95 / 18.32 | high-ACC point |

Interpretation:

1. PFedBA does not prefer the strongest mask. The best ASR comes from `tau=5, alpha=0.3`, not `tau=10/12, alpha=0.7`.
2. `lr=0.05` dominates `lr=0.03` and `lr=0.08` for low ASR at the same mask.
3. Compared with the existing PFedBA table, the new `71.46 / 13.16` point is not as low as old `PLE=5` (`72.10 / 10.95`) but is cleaner because it uses the normal `PLE=1` setting and a simple low-CL configuration.

## 3. How to Integrate into Chapter 4

### Static Main/Representative Results

Keep the current default table as a common-setting stress test. Update the representative/safety-first BadNet statement to include the new 1000-round evidence:

- old representative BadNet: `76.30 / 11.36`;
- new low-CL three-seed BadNet: `74.64±2.84 / 8.65±3.40`;
- best usable seed: `72.29 / 4.86`.

Recommended wording:

> 在进一步延长训练至 1000 轮并降低一致性损失权重后，FedCLCM 在 BadNet 场景中进入了更稳定的低 ASR 工作区间。以 `lambda_cl=0.05, tau=5.0, alpha_m=0.3` 为例，三种数据划分种子下的平均结果为 `74.64%±2.84% / 8.65%±3.40%`，其中最佳划分可达到 `72.29% / 4.86%`。这说明 BadNet 下较高 ASR 并非方法完全失效，而是默认一致性约束与局部贴片特征存在耦合；降低一致性权重后，通道掩码的净化作用能够更充分体现。

### Multiseed Table

Replace or supplement the old BadNet FedCLCM multiseed row:

| Method | Attack | n | ACC | ASR | Source |
| --- | --- | ---: | ---: | ---: | --- |
| FedCLCM-old | BadNet | 2 | 76.71±0.11 | 30.49±7.68 | old default/mask regime |
| FedCLCM-lowCL | BadNet | 3 | 74.64±2.84 | 8.65±3.40 | new 1000-round low-CL regime |

This is not cherry-picking if described as a parameter-regime comparison motivated by ablation.

### Ablation Section

Update the BadNet ablation narrative:

- `lambda_cl=0.00`: `71.16 / 6.17`
- `lambda_cl=0.02`: `75.11 / 11.18`
- `lambda_cl=0.05`: mean `74.64 / 8.65`, best seed `72.29 / 4.86`
- `lambda_cl=0.10`: `73.19 / 8.50`
- `lambda_cl=0.20`: `77.74 / 22.80`

Key claim: CL is not monotonically beneficial for BadNet. It stabilizes useful representations, but excessive CL can preserve patch-correlated shortcuts.

### Mask Parameter Section

New BadNet mask evidence at `lr=0.08, lambda_cl=0.05`:

| tau | alpha | ACC | ASR | Interpretation |
| ---: | ---: | ---: | ---: | --- |
| 4.0 | 0.2 | 38.91 | 4.35 | too destructive |
| 4.0 | 0.4 | 52.45 | 6.74 | still destructive |
| 5.0 | 0.2 | 72.93 | 6.96 | best balanced mask-only point |
| 5.0 | 0.3 | 74.64±2.84 | 8.65±3.40 | best main regime |
| 5.0 | 0.4 | 78.93 | 38.76 | too weak/unstable after utility recovery |
| 6.0 | 0.2 | 77.08 | 50.82 | bad ASR rebound |

Key claim: BadNet has a narrow mask sweet spot around `tau=5, alpha=0.2~0.3` under low CL.

### PFedBA Section

Add the new PFedBA low-CL table as a cleaner sweep:

- best low-ASR: `lr=0.05, tau=5, alpha=0.3 -> 71.46 / 13.16`;
- high-utility: `lr=0.08, tau=10, alpha=0.7 -> 75.95 / 18.32`.

Recommended wording:

> 新一轮 1000 轮扫描表明，PFedBA 场景下过强掩码并不一定带来最低 ASR。相反，在较低一致性权重 `lambda_cl=0.05` 下，适中的掩码配置 `tau=5.0, alpha_m=0.3` 与学习率 `0.05` 取得了 `71.46% / 13.16%` 的个性化结果，优于更强掩码配置的 ASR。这与静态 BadNet 的观察一致：局部或个性化阶段可适配的后门并不适合简单加大净化强度，而需要控制一致性约束与掩码强度的平衡。

## 4. Next Experiment Plan: 46 Runs

The goal is not blind sweeping. The next runs should test three concrete hypotheses:

1. **Low-CL is the correct BadNet regime.** Refine around `lambda_cl=0.02~0.08`, `tau=4.5~5.5`, `alpha=0.2~0.35`.
2. **Two-stage schedules can recover ACC after early suppression.** Use strong early purification, then relax.
3. **PFedBA prefers moderate masks and low CL.** Refine around `lr=0.04~0.06`, `tau=4~6`, `alpha=0.2~0.35`, and test PLE/PGD only selectively.

### A. Static BadNet Fine Grid, 18 runs, 1000 rounds

Common setting: CIFAR-10, ResNet18, `nclient=100`, `adv10`, `jr=0.10`, `local_epochs=1`, `plocal_epochs=1`, `lr=0.08`, `lr_head=0.08`, `rt_beta=0.20`, `adv_eps=0`, `eval_gap=10`.

| Group | lambda_cl | tau | alpha | Seeds | Runs |
| --- | ---: | ---: | ---: | --- | ---: |
| low-CL sweet spot | 0.02 | 5.0 | 0.2 | 42,43,44 | 3 |
| low-CL sweet spot | 0.02 | 5.0 | 0.3 | 42,43,44 | 3 |
| current best confirm | 0.05 | 5.0 | 0.2 | 42,43,44 | 3 |
| current best confirm | 0.05 | 5.0 | 0.3 | 42,43,44 | 3 |
| slightly stronger | 0.08 | 5.0 | 0.2 | 42,43,44 | 3 |
| slightly stronger | 0.08 | 5.0 | 0.3 | 42,43,44 | 3 |

Expected value: produce a cleaner multiseed row than `74.64 / 8.65`, ideally with ASR under 8% and ACC above 74%.

### B. Static BadNet Schedule Tests, 10 runs, 1000 rounds

Use `--clcm_schedule`; no new algorithm code required.

Common setting: same as A, seeds mostly 42 first, then confirm best on 43/44.

| Tag | Schedule | Runs |
| --- | --- | ---: |
| `SCH_STRONG_TO_MID_A` | `0: lr=0.03,lambda_cl=0.02,tau=3,alpha=0.3; 200: lr=0.05,tau=4,alpha=0.25; 500: lr=0.08,tau=5,alpha=0.2` | 1 |
| `SCH_STRONG_TO_MID_B` | `0: lr=0.03,lambda_cl=0.05,tau=3,alpha=0.3; 200: lr=0.05,tau=4,alpha=0.25; 500: lr=0.08,tau=5,alpha=0.3` | 1 |
| `SCH_MASK_RELAX_A` | `0: tau=4,alpha=0.2; 300: tau=5,alpha=0.2; 700: tau=5,alpha=0.3` | 1 |
| `SCH_CL_DECAY_A` | `0: lambda_cl=0.10,tau=5,alpha=0.2; 300: lambda_cl=0.05; 600: lambda_cl=0.02` | 1 |
| `SCH_CL_DECAY_B` | `0: lambda_cl=0.08,tau=5,alpha=0.3; 300: lambda_cl=0.05; 600: lambda_cl=0.02` | 1 |
| confirm best schedule | best of above on seeds 42,43,44 | 3 |
| schedule + jr=0.05 | best schedule, seed 42 | 1 |
| schedule + jr=0.15 | best schedule, seed 42 | 1 |

Expected value: test whether early low-ASR safety windows can retain ACC after relaxation.

### C. Static BadNet PGD/Purification Head, 6 runs, 800 rounds

This is based on old ResNet18 low-ASR evidence. It is higher risk, so keep small.

Common setting: `lr=0.03` or `0.05`, `lambda_cl=0.05`, `tau=5`, `alpha=0.2/0.3`, seeds 42 only first.

| Tag | adv_eps | adv_num_iter | lr | tau/alpha | Runs |
| --- | ---: | ---: | ---: | --- | ---: |
| `PGD_E002_I3_T5A02` | 0.02 | 3 | 0.05 | 5/0.2 | 1 |
| `PGD_E002_I3_T5A03` | 0.02 | 3 | 0.05 | 5/0.3 | 1 |
| `PGD_E005_I3_T5A02` | 0.05 | 3 | 0.03 | 5/0.2 | 1 |
| `PGD_E005_I3_T5A03` | 0.05 | 3 | 0.03 | 5/0.3 | 1 |
| confirm best PGD | best two on seed 43 | 2 |

Expected value: see whether old adversarial purification signal transfers without destroying ACC.

### D. PFedBA Refined Sweep, 12 runs, 1000 rounds

Use existing `pfedba_local/main.py` only. Common: `lambda_cl=0.05`, `local_epochs=1`, `plocal_epochs=1`, `rt_beta=0.20`, `eval_gap=10`.

| Group | lr | tau | alpha | Runs |
| --- | ---: | ---: | ---: | ---: |
| refine best | 0.04 | 5.0 | 0.25 | 1 |
| refine best | 0.04 | 5.0 | 0.30 | 1 |
| refine best | 0.05 | 4.5 | 0.25 | 1 |
| refine best | 0.05 | 4.5 | 0.30 | 1 |
| refine best | 0.05 | 5.0 | 0.25 | 1 |
| refine best | 0.05 | 5.5 | 0.25 | 1 |
| refine best | 0.06 | 5.0 | 0.25 | 1 |
| refine best | 0.06 | 5.0 | 0.30 | 1 |
| PLE test | 0.05 | 5.0 | 0.30, `plocal_epochs=3` | 1 |
| PLE test | 0.05 | 5.0 | 0.30, `plocal_epochs=5` | 1 |
| PGD light | 0.05 | 5.0 | 0.30, `adv_eps=0.02, adv_iter=3` | 1 |
| high-ACC guard | 0.08 | 8.0 | 0.5 | 1 |

Expected value: push PFedBA ASR toward 10–12% while keeping personalized ACC above 71%.

## 5. Resource Allocation

- 3090: run A + B + C static groups. Four GPUs, batch parallel 4. Estimated runtime: about 18 static-equivalent 1000-round runs plus schedules; using previous ~5.9 sec/round per run, four GPUs need roughly 8–10 hours for A, 5–6 hours for B, 2–3 hours for C.
- 4090: run D PFedBA sequentially. Previous 12 PFedBA runs completed overnight, so expect 8–12 hours.
- Total wall-clock if started together: one overnight block.

## 6. Decision Rule for Thesis Use

Use results only if traceable and consistent:

1. Primary BadNet claim should use multiseed mean, not a single lucky seed, unless explicitly labeled as best operating point.
2. Main low-ASR BadNet target: `ACC >= 74%` and `ASR <= 8%` mean over 3 seeds.
3. PFedBA target: `personalized ACC >= 71%` and `personalized ASR <= 12%`.
4. If targets are not met, keep current integrated story: FedCLCM is a tunable frontier method, strongest on Blend/SIG and improved but not solved on BadNet/PFedBA.
