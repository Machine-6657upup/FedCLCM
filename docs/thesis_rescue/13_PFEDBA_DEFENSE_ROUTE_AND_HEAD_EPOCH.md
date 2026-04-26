# PFedBA Defense Route And Head-Epoch Test

## Why This File Exists
Static stage3 is already the settled static source of truth. The next dynamic question is no longer:
- whether PFedBA can break the current line

That part is already locked.

The next useful questions are:
- which defense ideas are structurally more promising against PFedBA
- why those methods outperform the current FedRT / FedCLCM line
- which component is worth borrowing first
- whether `plocal_epochs` in official PFedBA FedRep materially changes attack effectiveness

## Locked Dynamic Starting Point
Authoritative PFedBA results must still come from:
- `/home/huangtu/PFL_Backdoor_Defense/PFedBA`

Current locked anchors:
- official PFedBA `FedRep` on CIFAR-10: personal clean about `0.7592`, personal ASR about `0.8336`
- official PFedBA `FedCLCM` on CIFAR-10: personal clean about `0.7480`, personal ASR about `0.5606`
- official PFedBA `FedRT` on CIFAR-10: best current personal clean about `0.7295`, best current personal ASR about `0.5475`

Working meaning:
- PFedBA clearly breaks `FedRep`
- `FedCLCM` and `FedRT` currently provide partial mitigation only
- neither is yet a thesis-grade final defense against PFedBA

## What Stronger PFedBA-Oriented Defenses Actually Do
The most relevant external defense lines currently identified are:
1. `BDPFL`
2. `PFL-ALP`

### 1. BDPFL
Direct relevance:
- this paper explicitly evaluates against `PFedBA`
- it is currently the most directly useful borrowing target

Core mechanism:
- layer-wise mutual distillation between the communication model and the personalized local model
- stronger emphasis on safer shallow/intermediate knowledge
- explanation-heatmap-guided intermediate distillation to suppress deeper backdoor features

Most important reported evidence:
- on CIFAR-10 under `PFedBA`, the paper reports:
- no defense: `ASR 100`, `ACC 85.78`
- `BDPFL`: `ASR 6.23`, `ACC 86.09`

Why this is more relevant than the current FedRT / FedCLCM line:
- it attacks the backdoor at the representation and activation-transfer level
- it does not rely mainly on server-side similarity trimming
- it preserves clean accuracy while reducing ASR, which is exactly the failure point of the current static FedCLCM story

Useful ablation facts already reported by the paper:
- removing explanation heatmap weakens both ASR suppression and ACC
- on CIFAR-10, the heatmap version is clearly better than the no-heatmap version
- larger distillation strength can keep reducing ASR, but too much hurts ACC

Practical judgment:
- if only one external defense family is borrowed first for PFedBA, `BDPFL` should be the first choice

### 2. PFL-ALP
Direct relevance:
- this paper does not directly report PFedBA
- but it explicitly studies gradient-aligned adaptive attacks, which is mechanistically close to PFedBA

Core mechanism:
- server-side dynamic clustering to reduce Non-IID bias and generate representative models
- client-side attention-based local purification to remove residual backdoor behavior

Most useful conceptual point:
- it uses clustering to mitigate heterogeneity, not as the final malicious detector
- this is exactly why it is more robust to gradient-aligned attacks than methods that treat cosine filtering as the main defense

Most useful reported ablation:
- clustering alone keeps higher clean accuracy but leaves higher ASR
- local purification alone reduces ASR but hurts clean accuracy
- the combination works best

Practical judgment:
- the most valuable component to borrow from `PFL-ALP` is the local purification logic
- its clustering idea is still useful, but should remain auxiliary rather than the main anti-PFedBA shield

## Why Current FedRT / FedCLCM Underperform
Current code-line tendency:
- most of the current logic still lives in the family of trim / cosine / consistency weighting / mild client perturbation

This is weaker against PFedBA because PFedBA is designed to:
- align the poisoned objective with the main objective
- reduce update-direction abnormality
- survive similarity-based or aggregation-side filtering

So the current line is structurally disadvantaged:
- `FedCLCM` still depends heavily on server-side robust aggregation and channel-level masking
- `FedRT` adds useful client-side robustness, but mainly through PGD-style head regularization plus augmentation
- neither currently contains a real representation-purification or teacher-student cleansing stage

Working conclusion:
- more cosine thresholds and more trim variants are unlikely to be the decisive fix
- the next real gain is more likely to come from client-side purification / distillation / activation alignment

## Best Borrowing Priority
Current recommendation, in order:

1. `BDPFL-lite` inside official `PFedBA/`
- keep the existing `FedRep` split structure
- add a benign-client purification path instead of building a new platform
- start from layer-wise distillation and only then consider explanation-heatmap refinement

2. `PFL-ALP-lite` ideas as support
- keep clustering as a heterogeneity helper
- borrow local purification logic
- do not treat clustering itself as the main defense story

3. Keep current `FedRT` components only as auxiliary
- PGD head training
- trigger-breaking augmentation
- optional contrastive consistency

The intended final combined direction is:
- `FedRep` communication structure
- benign-side local purification / distillation from `BDPFL`
- optional light clustering or representative-model aid from `PFL-ALP`
- current `FedRT` robustness tricks kept as secondary helpers, not the main thesis novelty

## Code-Availability Note
As of `2026-04-18`, the project-side quick search did not confirm an official public GitHub implementation for:
- `BDPFL`
- `PFL-ALP`

So the practical route should be:
- first borrow mechanisms from the papers
- implement a minimal version inside the authoritative `PFedBA/` code line
- only pivot to full external reproduction if official source code is later found

## Why `plocal_epochs` Is Worth Testing In FedRep
In official PFedBA `FedRep`, client training is explicitly split:
- head phase: `plocal_epochs`
- base phase: `local_epochs`

This matters even though only the shared base is aggregated.

Reason:
- the poisoned head phase changes the local classifier before base training starts
- then the poisoned base phase is optimized while the head is frozen
- so a stronger head phase can change the gradient signal that the shared base receives
- therefore `plocal_epochs` can plausibly change final PFedBA ASR, not just clean adaptation

This is consistent with the code in:
- `PFedBA/main.py`
- `PFedBA/FLAlgorithms/users/userrep.py`

## Minimal FedRep Head-Epoch Test
Use the official PFedBA FedRep CIFAR-10 anchor setting and vary only:
- `plocal_epochs`

Keep fixed:
- `dataset = Cifar10`
- `model = resnet`
- `algorithm = FedRep`
- `learning_rate = 0.1`
- `lr_head = 0.1`
- `local_epochs = 20`
- `num_global_iters = 150`
- `numusers = 10`
- `batch_size = 64`
- `attack_start = 30`
- `attack_method = attackall`
- `poisoning_per_batch = 5`
- `defense = none`
- `per_epoch = 1`
- `malclient = 10`

Recommended sweep:
- `plocal_epochs = 1`
- `plocal_epochs = 2`
- `plocal_epochs = 5`
- `plocal_epochs = 10`

Optional extreme point later:
- `plocal_epochs = 20`

Primary metrics to compare:
- final personalized clean accuracy
- final personalized ASR
- global clean accuracy
- global ASR

Interpretation rule:
- do not treat lower ASR as a defense gain if clean accuracy also collapses

## Expected Runtime
Recent official `FedRT + PFedBA` CIFAR-10 runs provide a practical timing reference:
- about `96` to `102` minutes per run
- about `100` minutes per run on average

So this 4-point FedRep head-epoch sweep should take:
- about `1 hour 45 minutes` wall clock on `4` GPUs if run fully in parallel

## Practical Next Action
The most useful next dynamic step is:
1. run the 4-point `FedRep plocal_epochs` sweep in official `PFedBA/`
2. confirm whether head-phase length is a real PFedBA amplifier
3. if yes, lock that finding into the thesis narrative
4. then start a minimal `BDPFL-lite` implementation inside official `PFedBA/`

## Revised 12-Hour Recommendation
The earlier large head-phase grid is not the best next use of time.

The better use of a 12-hour window is:
- small component-validation experiments
- not broad hyperparameter sweeping

Reason:
- the thesis bottleneck is no longer "can we search enough knobs"
- the real bottleneck is "which borrowed defense component actually changes the PFedBA outcome"

Using the actual runtime reference from the official `2026-04-18` `FedRT + PFedBA` batch:
- single run: about `100` minutes
- stable mode: `1` process per GPU

So the practical batch size is:
- about `8` runs if we want two waves with large safety margin
- about `10` runs if all jobs are clean and stable

## What Should Be Tested First
Priority should be on component value, in this order:

1. benign-side local purification
- borrowed from `PFL-ALP`
- this is the highest-priority component to validate first
- it attacks residual backdoor behavior inside the local model, not just aggregation anomalies

2. layer-wise mutual distillation
- borrowed from `BDPFL`
- this is the highest-priority representational component
- it is more aligned with PFedBA than more trim/cosine variants

3. explanation / attention-map alignment
- also borrowed from `BDPFL`
- useful, but should be tested after confirming that plain purification / distillation already helps

4. tiny `plocal_epochs` confirmation only
- keep this as a very small side test
- do not let it dominate the 12-hour window

## New Method Implemented In Official PFedBA
As of `2026-04-18`, a new paper-grounded lite method has been added to the official PFedBA code line:
- algorithm name: `FedRPD`

Code locations:
- `/home/huangtu/PFL_Backdoor_Defense/PFedBA/FLAlgorithms/users/userfedrpd.py`
- `/home/huangtu/PFL_Backdoor_Defense/PFedBA/FLAlgorithms/servers/serverfedrpd.py`
- `/home/huangtu/PFL_Backdoor_Defense/PFedBA/main.py`

Design choice:
- keep the current `FedRT` communication and trimmed aggregation path
- keep the current benign-side PGD head hardening and trigger-breaking augmentation
- add `PFL-ALP`-style benign local purification
- add `BDPFL`-style layer-wise feature distillation

## New Full-Reproduction Branch In Official PFedBA
As of `2026-04-18`, a first full-reproduction-oriented `PFL-ALP` branch has also been added directly inside the official PFedBA workspace:
- algorithm name: `PFLALP`

Code locations:
- `/home/huangtu/PFL_Backdoor_Defense/PFedBA/FLAlgorithms/users/userpflalp.py`
- `/home/huangtu/PFL_Backdoor_Defense/PFedBA/FLAlgorithms/servers/serverpflalp.py`
- `/home/huangtu/PFL_Backdoor_Defense/PFedBA/main.py`

Current implementation scope:
- `PFL-ALP`-style server-side dynamic clustering on selected-client shared-base updates
- cluster-wise representative model construction
- benign-client attention-based local purification with `NAD + CE`
- purified representative stored as the client personalized model for later inference

Important evaluation fix:
- the official PFedBA code had a dormant bug in `evaluate_personalized_model()`
- it wrote into nonexistent fields `rs_glob_acc_per / rs_train_acc_per / rs_train_loss_per`
- this was corrected to the actually initialized personalized-result buffers
- personalized-model poison evaluation was also added, because the stock PFedBA code only had clean personalized evaluation

Practical meaning:
- `PFLALP` can now be tested in the official PFedBA attack workspace
- and its logs can now report two different metric views:
- legacy `one-step` PFedBA personalized metrics
- saved personalized-model metrics, which are the more faithful `PFL-ALP` inference metrics

## PFLALP Minimal Smoke Result
Minimal smoke setting used on `2026-04-18`:
- `dataset = Cifar10`
- `algorithm = PFLALP`
- `model = resnet`
- `learning_rate = 0.1`
- `lr_head = 0.1`
- `plocal_epochs = 1`
- `local_epochs = 20`
- `num_global_iters = 0`
- `numusers = 10`
- `batch_size = 64`
- `attack_start = 30`
- `poisoning_per_batch = 5`
- `purify_beta = 1500`
- `purify_rounds = 1`
- `cluster_max_k = 4`

Smoke-test conclusion:
- the full `PFLALP` code path runs through in the official PFedBA repository
- dynamic clustering executes successfully
- benign personalized purification executes successfully
- saved personalized-model evaluation also runs successfully after the evaluation-path fix

Smoke metrics are only sanity anchors, not scientific conclusions:
- final global clean about `0.1222`
- final global ASR about `0.0046`
- legacy one-step personal clean about `0.2786`
- legacy one-step personal ASR about `0.0086`
- saved personalized-model clean about `0.1107`
- saved personalized-model ASR about `0.0063`

Interpretation:
- this confirms code-path correctness only
- it does not yet show whether `PFLALP` really defends PFedBA under converged settings
- real judgment still requires the planned multi-round runs

## 2026-04-18 Midday Correction: Stop Sweeping `local_epochs`
The latest official PFedBA batch made one point clearer:
- sweeping `local_epochs` itself is no longer the right next axis

Reason:
- with `local_epochs = 10`, `FedRep` at `150` global rounds still showed late-round upward movement in clean accuracy
- example late rounds:
- round145: `gacc 0.781`, `gasr 0.809`
- round146: `gacc 0.770`, `gasr 0.803`
- round147: `gacc 0.777`, `gasr 0.902`
- round148: `gacc 0.777`, `gasr 0.841`
- round149: `gacc 0.780`, `gasr 0.780`
- round150: `gacc 0.793`, `gasr 0.792`

Working judgment:
- `150` rounds is not yet a trustworthy convergence point for the `local_epochs = 10` line
- therefore the next batch should fix:
- `local_epochs = 10`
- `plocal_epochs = 1`
- and only increase `num_global_iters`

Immediate practical batch after this correction:
- compare `FedRep`, `FedRT`, and `FedRPD` component variants under fixed `local_epochs = 10`
- use larger global-round checkpoints first, currently `250` and `300`
- do not claim any method works just because ASR drops if clean accuracy also collapses

Implementation note:
- prepared launcher:
- `/home/huangtu/PFL_clean_workspace/root_static/scripts/run_pfedba_fixedle10_global12_nohup.sh`
- prepared worker:
- `/home/huangtu/PFL_clean_workspace/root_static/scripts/pfedba_fixedle10_global12_worker.sh`

## 2026-04-18 Hyperparameter Interpretation Fix
Recent official PFedBA logs make three implementation facts clear.

### 1. `rt_beta` Is Discrete Under `numusers = 10`
In current official `PFedBA/`:
- `FedRT` and `FedRPD` both aggregate with:
- `k = int(rt_beta * n)`
- where `n = len(selected_users)`

With the current standard regime:
- `numusers = 10`

So these settings are effectively identical:
- `rt_beta = 0.00`
- `rt_beta = 0.05`
- `rt_beta = 0.08`

Because all of them produce:
- `k = 0`

Likewise these are also effectively identical:
- `rt_beta = 0.10`
- `rt_beta = 0.12`
- `rt_beta = 0.15`

Because all of them produce:
- `k = 1`

Practical consequence:
- small decimal sweeps around `0.1` are mostly fake sweeps in the current `10`-user aggregation regime
- future `rt_beta` search should be described by effective trim count `k`, not by raw decimal alone

### 2. What The Recent `FedRT` 12-Run Batch Actually Says
Authoritative log directory:
- `/home/huangtu/PFL_Backdoor_Defense/PFedBA/log/fedrt_nohup12_20260418_025933`

Locked reading:
- removing PGD head hardening hurts badly
- removing trimmed aggregation hurts badly
- removing trigger-breaking augmentation also hurts
- the current best balanced point in that batch remains the anchor:
- `rt_beta = 0.10`
- `adv_eps = 0.10`
- `adv_num_iter = 5`
- `aug_strength = 0.10`

Its final personalized result is about:
- clean acc `0.7295`
- ASR `0.5475`

This is only partial mitigation, not a thesis-final solution.

### 3. The Next Useful Batch Should Be Module-Oriented, Not Decimal-Oriented
Therefore the next stable dynamic batch should fix:
- `dataset = Cifar10`
- `model = resnet`
- `learning_rate = 0.1`
- `lr_head = 0.1`
- `batch_size = 64`
- `plocal_epochs = 1`
- `local_epochs = 10`
- `num_global_iters = 300`
- `numusers = 10`
- `attack_start = 30`
- `poisoning_per_batch = 5`
- `per_epoch = 1`
- `malclient = 10`
- `rt_beta = 0.10`

And only compare:
- `FedRep` baseline
- `FedRT` anchor
- `FedRPD` purification only
- `FedRPD` distillation only
- `FedRPD` full

Prepared simple launcher:
- `/home/huangtu/PFL_clean_workspace/root_static/scripts/run_pfedba_module_repro5_r300_nohup.sh`

## 2026-04-18 Fidelity Audit: Are Purification And Distillation Really Reproduced?
Short answer:
- `PFL-ALP` purification is only a partial but structurally aligned transplant
- `BDPFL` distillation is only a lite borrowing, not a strict reproduction

### What Is Correctly Aligned With `PFL-ALP`
Paper-side locked points:
- benign-side attention-based purification
- teacher-student purification with task loss plus NAD loss
- attention map power `p = 2`
- CIFAR-10 benign-side setting: `E = 1`, `B = 64`, `lr = 0.1`
- ablation-best purification strength around `beta = 1500`

Current code alignment in:
- `/home/huangtu/PFL_Backdoor_Defense/PFedBA/FLAlgorithms/users/userfedrpd.py`

What is aligned:
- uses NAD-style normalized attention matching
- uses task loss + `purify_beta * NAD`
- uses `purify_beta = 1500` by default
- uses `purify_rounds = 1` by default
- purification optimizer currently uses `learning_rate`, so with the current anchor it is `0.1`

What is not a strict reproduction:
- the paper first updates the local personalized model in the same purification round, then uses it as a frozen teacher for the received representative model
- current code instead keeps a carried-over `personal_teacher` from the previous round and purifies the current model against that stored teacher
- the paper also includes server-side dynamic clustering and representative-model generation, which the current PFedBA-lite transplant does not implement

Practical judgment:
- this is fair to call `PFL-ALP-lite`
- it is not fair to call it an exact reproduction of `PFL-ALP`

### What Is Correctly Aligned With `BDPFL`
Paper-side locked points:
- two-model personalized setting: communication model plus local personalized model
- mutual distillation with CE + KL in both directions
- layer-wise weight decay:
- `beta_l = 1 / (1 + gamma * l)`
- `gamma = 1` is already a satisfactory setting
- explanation-heatmap distillation is an explicit important component
- CIFAR-10 outer setting: `T = 1000`, `E = 20`, `lr = 0.1`
- distillation-epoch ablation varies roughly `10 -> 50`

Current code alignment:
- keeps layer-wise decay controlled by `distill_gamma`
- current default `distill_gamma = 1.0`
- adds weighted intermediate feature matching on several ResNet stages

What is not a strict reproduction:
- current code uses one-way feature MSE from teacher to student
- it does not implement the paper’s bidirectional CE + KL mutual distillation
- it does not implement Grad-CAM / explanation-heatmap distillation
- it does not expose a separate multi-epoch distillation stage like the paper’s distillation-epoch study
- current `distill_weight = 1.0` is a neutral engineering default, not a scalar directly locked by the paper text

Practical judgment:
- this is fair to call `BDPFL-lite feature distillation`
- it is not fair to describe it as a strict `BDPFL` reproduction

### Correct Wording For The Thesis And Logs
Use:
- `PFL-ALP-inspired local purification`
- `BDPFL-inspired layer-wise feature distillation`
- `paper-grounded lite transplant`

Do not use:
- `we reproduced PFL-ALP`
- `we reproduced BDPFL`

### Immediate Experiment Implication
The current 5-run module batch is still useful because it answers:
- whether `PFL-ALP`-style purification helps in the official PFedBA platform
- whether `BDPFL`-style feature distillation helps in the official PFedBA platform
- whether the two lite components are complementary

But it does not answer:
- whether full `PFL-ALP`
- or full `BDPFL`
- can be reproduced as-is in this codebase

This is intentionally a `lite` adaptation rather than a claim of exact reproduction.

Reason:
- no official public source code was confirmed for `BDPFL` or `PFL-ALP`
- therefore the correct engineering move is to implement a minimal, paper-grounded version inside the authoritative `PFedBA/` platform

## FedRPD Default Hyperparameters And Their Sources
The default `FedRPD` settings are intentionally sparse.

### Inherited from current best FedRT anchor
- `rt_beta = 0.10`
- `adv_eps = 0.10`
- `adv_num_iter = 5`
- `aug_strength = 0.10`

Reason:
- these are the best current official PFedBA `FedRT` anchor values in this project line

### Borrowed from PFL-ALP
- `purify_beta = 1500`
- `purify_rounds = 1`

Reason:
- `PFL-ALP` ablation identifies `β = 1500` as the best tradeoff for CIFAR-10
- `purify_rounds = 1` is the conservative default because the paper uses `E = 1` benign local training and the current PFedBA platform is already much heavier (`local_epochs = 20`)

### Borrowed from BDPFL
- `distill_gamma = 1.0`
- `distill_weight = 1.0`

Reason:
- `BDPFL` reports that `γ = 1` is already sufficient for a satisfactory defense effect
- the visible text does not give a clearly locked extra scalar for the layer-wise feature term, so `distill_weight = 1.0` is used as the neutral default

### Architectural simplification
- feature layers used: `layer1` to `layer4` inside the CIFAR ResNet `base`

Reason:
- these are the natural intermediate representation layers after the FedRep split in the official PFedBA code
- they are also the most meaningful layers for attention / feature purification without changing the network structure

## What FedRPD Actually Optimizes
For benign clients only:

1. keep the existing `FedRT` head phase
- optional PGD head hardening
- local head update

2. keep the existing `FedRT` shared-base phase
- benign trigger-breaking augmentation

3. add a purification phase after local training
- teacher: previous local personalized snapshot
- student: current local model
- task loss: clean CE on local data
- purification loss: `PFL-ALP`-style NAD attention alignment
- distillation loss: `BDPFL`-style layer-wise feature MSE with shallow-to-deep decay

Practical meaning:
- this directly targets the representational persistence that PFedBA tries to exploit
- it is not just another aggregation-side filter

## First Validation Script
Prepared launcher:
- `/home/huangtu/PFL_clean_workspace/root_static/scripts/run_pfedba_fedrpd_component_12.sh`

Purpose:
- compare `FedRep`, `FedRT`, and `FedRPD`
- isolate whether purification or layer-wise distillation contributes more
- keep the batch focused on component value rather than broad hyperparameter sweeping

## Recommended 8-Run Batch
Assuming a minimal `BDPFL-lite / PFL-ALP-lite` implementation is added inside official `PFedBA/`, the best 8-run batch is:

### Group A: locked anchors
1. `FedRep` baseline
2. `FedRT` current best anchor

Purpose:
- keep a same-night comparison anchor

### Group B: single borrowed components
3. `FedRep + local purification`
4. `FedRep + layer-wise distillation`
5. `FedRT + local purification`
6. `FedRT + layer-wise distillation`

Purpose:
- directly answer which single borrowed component is genuinely useful

### Group C: combined variants
7. `FedRep + local purification + layer-wise distillation`
8. `FedRT + local purification + layer-wise distillation`

Purpose:
- test whether the two strongest components are complementary

## Optional 10-Run Version
If there is enough time after the 8 core runs, add only:
9. `FedRep` with `plocal_epochs = 5`
10. `FedRep + best borrowed component` with `plocal_epochs = 5`

Purpose:
- confirm whether head-phase strength materially changes the attack and whether the borrowed defense remains effective there

## What Should Not Be The Main Batch
Do not spend the main 12-hour batch on:
- large `plocal_epochs × lr_head` grids
- more cosine-gate threshold sweeps
- more trim-ratio sweeps
- server-side clustering as the primary defense story

Those are secondary refinements, not the current thesis bottleneck.
