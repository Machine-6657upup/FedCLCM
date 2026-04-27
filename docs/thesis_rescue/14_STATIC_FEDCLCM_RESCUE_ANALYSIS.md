# Static FedCLCM Rescue Analysis - 2026-04-27

## Bottom Line

FedCLCM/CLCM did achieve `ACC >= 75%` and `ASR < 10%` in old 3090 logs, but the good CIFAR-10 results are concentrated in an old regime:

- `Cifar10`, `nclient=40`, usually `adv5`, `ResNetP`, `global_rounds=800`
- often `lr=0.003`, `lr_head=0.01`, `local_epochs=1`, `join_ratio=1.0`
- an alternative good family uses `lr=0.1`, `lr_head=0.05`, `join_ratio=0.25`, `plocal_epochs=2`, `adv_eps=0.03`, `adv_num_iter=10`, `rt_beta=0.2`, weak/no mask

The current formal static setting is different:

- `Cifar10`, `nclient=100`, `adv10`, `ResNet18`, `global_rounds=600`
- `join_ratio=0.1`, `lr=0.1`, `lr_head=0.1`, `local_epochs=1`, `plocal_epochs=1`

I did not find an old CIFAR-10 FedCLCM/CLCM log with `nclient=100` that also has `ACC >= 75%` and `ASR < 10%`. Old `nclient=100` low-ASR FedCLCM logs are mainly FashionMNIST/CNN, not CIFAR-10/ResNet.

This means the current bad static results are not well explained by "the FedCLCM code was broken". The old and new `User/clientCLCM.py` and `User/serverCLCM.py` core files are effectively the same. The gap is primarily setting transfer: old successful CLCM was tuned for a smaller, pretrained-ResNetP, lower-lr or head-PGD regime; current formal static uses the paper-common 100-client/10-selected/high-lr/non-pretrained regime.

## Confirmed Old CIFAR-10 Success Logs

| Log | Setting | Final ACC | Final ASR | Key Params |
| --- | --- | ---: | ---: | --- |
| `thesis_log/attack_generalization_static/sig_fedclcm_20260408_222651.log` | `Cifar10_dir0.5_bdoor0.2_nclient_40_sig_adv5` | 0.7751 | 0.0213 | `ResNetP`, `jr=1.0`, `lr=0.003`, `lr_head=0.01`, `rounds=800`, `lambda_cl=0.1`, `mask_tau=10`, `mask_alpha=0.9` |
| `logs/fedclcm_pgd_bestguess_badnet_adv5/PGD_weakmask_beta02_gpu1.log` | `Cifar10_dir0.5_bdoor0.2_nclient_40_badnet_adv5` | 0.7979 | 0.0269 | `ResNetP`, `jr=0.25`, `lr=0.1`, `lr_head=0.05`, `plocal_epochs=2`, `adv_eps=0.03`, `adv_iter=10`, `rt_beta=0.2`, weak mask |
| `logs/fedclcm_pgd_bestguess_badnet_adv5/PGD_nomask_beta02_gpu0.log` | `Cifar10_dir0.5_bdoor0.2_nclient_40_badnet_adv5` | 0.8109 | 0.0322 | same as above, no effective mask |
| `thesis_log/scalability_robustness/adv_5_20260413_150619.log` | `Cifar10_dir0.5_bdoor0.2_nclient_40_badnet_adv5` | 0.7734 | 0.0446 | `ResNetP`, `jr=1.0`, `lr=0.003`, `lr_head=0.01`, `rt_beta=0.05`, `lambda_cl=0.2`, `mask_tau=12`, `mask_alpha=0.7` |
| `thesis_log/main_results_basic/cifar_fedclcm_20260325_124727.log` | `Cifar10_dir0.5_bdoor0.2_nclient_40_badnet_adv5` | 0.7706 | 0.0455 | `ResNetP`, `jr=1.0`, `lr=0.003`, `lr_head=0.01`, `rt_beta=0.0`, `lambda_cl=0.1`, `mask_tau=10`, `mask_alpha=0.9` |
| `thesis_log/badnet_fedclcm_training_hp/hp_jr_0p25_20260414_231453.log` | `Cifar10_dir0.5_bdoor0.2_nclient_40_badnet_adv5` | 0.8008 | 0.0493 | `ResNetP`, `jr=0.25`, `lr=0.003`, `lr_head=0.01`, `rt_beta=0.05`, `lambda_cl=0.2`, `mask_tau=12`, `mask_alpha=0.7` |
| `thesis_log/badnet_fedclcm_training_hp/hp_ls_2_20260414_231453.log` | `Cifar10_dir0.5_bdoor0.2_nclient_40_badnet_adv5` | 0.7981 | 0.0467 | `ResNetP`, `jr=1.0`, `lr=0.003`, `lr_head=0.01`, `local_epochs=2`, `rt_beta=0.05`, `lambda_cl=0.2` |

Important interpretation:

- The `lr=0.003/lr_head=0.01/ResNetP/800-round` family gives stable low ASR around 4%-6% with ACC around 77%-80%.
- The `head-PGD + rt_beta=0.2 + no/weak mask` family gives stronger badnet results, including `ACC=0.8109, ASR=0.0322`.
- These are real logs, not memory. But they are not aligned to the current formal `nclient=100, adv10, jr=0.1, ResNet18` setting.

## Current Formal Static Results

From `thesis_formal_logs/final_static_main_20260424/static_summary_tmp.csv`:

| Attack | FedAvg ACC/ASR | FedRep ACC/ASR | FedCLCM ACC/ASR |
| --- | --- | --- | --- |
| Badnet | 0.513 / 0.934 | 0.787 / 0.575 | 0.757 / 0.522 |
| Blend | 0.525 / 0.711 | 0.792 / 0.179 | 0.752 / 0.160 |
| SIG | 0.497 / 0.993 | 0.791 / 0.339 | 0.749 / 0.311 |

This is usable to show static attacks are not weak anymore, because FedRep is clearly attacked. But it is not enough to claim FedCLCM is a strong static defense, especially for Badnet/SIG.

## Current 12h Static Sweep Result

From `runs/20260427_12h_4090_static_final_confirm/summary.csv`:

- `B04_600`: `ACC=0.7897`, `ASR=0.2785`
- `B07_600`: `ACC=0.7749`, `ASR=0.4912`

From `runs/20260427_12h_3090_localepoch_badnet_limit/summary.csv`:

- Best low-ASR badnet candidate: `BN_LIM07`, `ACC=0.7287`, `ASR=0.1191`
- Best high-ACC badnet candidates remain around `ACC=0.76-0.77`, but `ASR=0.39-0.48`
- Local epochs lower ASR but hurt ACC:
  - Badnet `E=1`: `ACC=0.7681`, `ASR=0.4502`
  - Badnet `E=10`: `ACC=0.6952`, `ASR=0.1940`
  - Badnet `E=20`: `ACC=0.6457`, `ASR=0.2038`
  - Blend shows a clearer ASR decrease, but ACC collapses as `E` grows.
  - SIG has unstable tradeoff; `E=10/20` lowers ASR but ACC is too low.

Conclusion: static FedCLCM cannot be rescued sufficiently by only sweeping `rt_beta/lambda_cl/aug/mask/local_epoch` in the current implementation.

## Why The Gap Is So Large

The most defensible explanation is a combination of four setting changes:

1. `ResNetP` vs non-pretrained `ResNet18`
   Old good logs use `ResNetP`. Current formal uses non-pretrained `ResNet18`. This directly affects representation quality and clean ACC.

2. `nclient=40/adv5` vs `nclient=100/adv10`
   Both are 10%-12.5% malicious ratios, but the selected-client dynamics differ. In old `jr=1.0`, every round sees all clients. In current `jr=0.1`, only 10 clients are selected, and malicious participation is stochastic unless the data generator/selection logic pins attackers.

3. `lr/lr_head`
   Old stable CLCM uses `lr=0.003`, `lr_head=0.01`; current formal uses `lr=0.1`, `lr_head=0.1`. The old PGD family uses `lr=0.1`, but compensates with `lr_head=0.05`, `plocal_epochs=2`, and `adv_eps=0.03/iter=10`.

4. Defense mechanism is still mostly server-side update filtering
   FedCLCM relies on channel mask, trimmed mean, cosine gating/layer trim, and client contrastive augmentation. Static and PFedBA-style attacks can survive if the poisoned representation is not cleaned inside the benign client's personalized model.

## Paper Evidence

BDPFL (`paper_extract/2503.06554v1.txt`) supports the target behavior we want:

- It uses 100 clients, 10 selected clients, `lr=0.1`, `E=20`, CIFAR-10 with ResNet-18, and learning-rate decay.
- It reports strong static results in the appendix table:
  - Blend: non-defense `ASR=86.12, ACC=71.74`; BDPFL `ASR=3.74, ACC=79.95`
  - SIG: non-defense `ASR=71.99, ACC=73.63`; BDPFL `ASR=1.24, ACC=80.17`
  - WanNet/Hidden also reach around 80% ACC with very low ASR.
- Its main idea is not another trim rule. It uses layer-wise mutual distillation plus explanation heatmap/attention transfer.

PFL-ALP (`paper_extract/PFL-ALP_pdf.txt`) explains why simply strengthening similarity filtering is the wrong main line:

- Server-side clustering is used to mitigate Non-IID heterogeneity and generate representative models, not as the final malicious detector.
- The actual defense is client-side attention-based local purification using personalized model knowledge.
- Its experiment setting is also different from our failed quick reproduction: CIFAR-10 ResNet18, 100 clients, 10 selected per round, 30% malicious in the paper's main setup, benign `E=1/lr=0.1`, adversarial `E=6/lr=0.05`.

## Why Our PFL-ALP / BDPFL Reproduction Was Not Conclusive

The existing scripts are not faithful enough to conclude that these papers "do not work":

- `scripts/run_pfedba_pflalp_full_r300.sh` runs inside the PFedBA codebase with `numusers=10`, `malclient=10`, `attackall`, `resnet_pretrained=0`, `local_epochs=10`, `purify_rounds=1`, `purify_beta=1500`.
- The 3090 PFL-ALP logs under `PFedBA/log/pflalp_full_r300_*` are incomplete or very short. Two full logs are empty, one stops around round 6, and the debug run uses `num_global_iters=0`.
- The debug PFL-ALP result has very low ACC, so it is an implementation/setting failure, not a meaningful negative result against the paper.
- I did not find a completed BDPFL reproduction summary in the same log location.

Therefore, the correct conclusion is: previous reproduction attempts were useful as exploration, but they are not valid evidence that BDPFL/PFL-ALP ideas fail.

## Short-Term Thesis Decision

Do not abandon FedCLCM immediately, but do not keep pretending the current static table is enough.

The practical path is:

1. Keep current formal static table as "FedRep is attackable; current FedCLCM helps on Blend/SIG but not enough on Badnet/SIG".
2. Run a focused old-success transfer test:
   - `nclient=100, adv10, ResNetP, lr=0.003, lr_head=0.01, rounds=800`
   - `nclient=100, adv10, ResNetP, lr=0.1, lr_head=0.05, plocal_epochs=2, adv_eps=0.03, adv_iter=10, rt_beta=0.2, no/weak mask`
   - Keep `nclient=40, adv5` as a positive-control sanity run.
3. If old-success transfer reaches `ACC >= 0.76` and `ASR <= 0.10`, use this as the static rescue configuration and document that the current method is sensitive to model/optimization regime.
4. If transfer still fails under `nclient=100`, stop spending time on more server-side mask sweeps.

## Method Upgrade If Transfer Fails

The minimal defensible upgrade is not "FedCLCM + more trim". It should be a FedCLCM-compatible client-side purification module:

- Keep FedCLCM server aggregation as baseline.
- On benign clients, maintain a clean personalized teacher snapshot.
- After receiving the global/shared base, perform a short local purification phase on clean local data.
- Add feature/attention alignment loss between the current model and the clean teacher, plus normal CE.
- Start with one or two layers only to avoid large code risk:
  - mid/high feature map attention loss for ResNet blocks
  - `purify_beta` small grid
  - `purify_rounds=1`
- Evaluate first on Badnet because it is the hardest current static case.

This is the closest thesis-safe version of borrowing from BDPFL/PFL-ALP: it attacks the actual weakness, which is poisoned personalized representation, instead of adding another server-side similarity threshold.

## Next Experiments To Run

Priority order:

1. Static transfer-positive-control:
   - `nclient=40, adv5, ResNetP`, old `lr=0.003/lr_head=0.01`, badnet/sig, 800 rounds.
   - Purpose: verify current cleaned repo can reproduce old good CLCM.

2. Static old-recipe transfer to paper-common scale:
   - `nclient=100, adv10, ResNetP`, same low-lr recipe.
   - Purpose: test whether old low ASR survives current scale.

3. Static head-PGD transfer:
   - `nclient=100, adv10, ResNetP`, `lr=0.1/lr_head=0.05`, `plocal_epochs=2`, `adv_eps=0.03`, `adv_iter=10`, `rt_beta=0.2`, no/weak mask.
   - Purpose: test the strongest old badnet recipe.

4. If 1 succeeds but 2/3 fail:
   - implement FedCLCM-Purify-lite.
   - do not spend more 12h sweeps on mask-only hyperparameters.

