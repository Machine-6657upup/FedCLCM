# PFedBA And Literature

## Why PFedBA Became The Hard Case
PFedBA is not just "another strong attack". It specifically stresses a weakness of methods that rely on:
- update similarity
- cosine filtering
- trim-like anomaly filtering
- aggregation-side outlier assumptions

This is why PFedBA feels much more painful for FedCLCM-style logic than ordinary static attacks.

## Important Strategic Clarification
PFedBA is a real long-term challenge, but it should not currently dominate the rescue schedule.

Reason:
- if the static baseline is still unstable, PFedBA only multiplies confusion
- the thesis needs a coherent core line first
- static rescue is the current foundation task

## Recovered Recent PFedBA Log Anchor

After static stage3 was consolidated, the next useful PFedBA anchor was recovered from the historical repository rather than from memory.

Important source-scope rule:
- for PFedBA code interpretation, only `/home/huangtu/PFL_Backdoor_Defense/PFedBA` and its directly related logic should be treated as authoritative
- historical PFedBA-like code outside that GitHub-source subfolder should not be used as evidence because it was later generated and is not trusted
- therefore all mechanism judgments below are intentionally restricted to the PFedBA GitHub-source subtree and the logs/results produced from that line

Most useful recent FedRep log:
- `/home/huangtu/PFL_Backdoor_Defense/logs/pfedba_cifar10_quad_20260416_165930/gpu2_fedrep_rn0.log`

Most useful recent FedRep result directory:
- `/home/huangtu/PFL_Backdoor_Defense/PFedBA/results/Cifar10_Apr.16_16.59.33_FedRep_attackall_none_5_1`

Matching recent FedCLCM result directory for comparison:
- `/home/huangtu/PFL_Backdoor_Defense/PFedBA/results/Cifar10_Apr.16_16.59.33_FedCLCM_attackall_none_5_1`

### What This Recent FedRep PFedBA Run Shows

This run is strong enough to lock the qualitative conclusion:
- PFedBA definitely can break FedRep in this project line
- this is not just an old-memory claim

Final personalized metrics at epoch `150` from the recovered FedRep result files:
- personal clean acc: about `0.7592`
- personal ASR: about `0.8336`
- global clean acc: about `0.7670`
- global ASR: about `0.8276`

These values come from:
- `per_test_result.csv`
- `per_posiontest_result.csv`
- `global_test_result.csv`
- `global_posiontest_result.csv`
under the result directory above

Matching recent FedCLCM comparison at epoch `150`:
- personal clean acc: about `0.7480`
- personal ASR: about `0.5606`
- global clean acc: about `0.7586`
- global ASR: about `0.5276`

### Hyperparameters Of The Recovered FedRep Log

The recovered FedRep log itself explicitly shows:
- `Algorithm = FedRep`
- `Dataset = Cifar10`
- `Local Model = resnet`
- `ResNet Pretrained = 0`
- `learning_rate = 0.1`
- `batch_size = 64`
- `numusers = 10`
- `global rounds = 150`
- `local rounds = 20`
- `attack_start = 30`
- `per_local epoch = 1`
- `defense = none`

The PFedBA code path also uses:
- `poisoning_per_batch = 5` by default in `PFedBA/main.py`
- the result directory suffix `_5_1` is consistent with `poisoning_per_batch = 5` and `per_epoch = 1`

### Clarification On `local_epochs = 20`

This `20` should be described carefully.

What is solidly supported:
- the recovered Apr-16 FedRep PFedBA log explicitly uses `local rounds = 20`
- the PFedBA repository parser default is `local_epochs = 20`
- the PFedBA README example commands also use `local_epochs 20`

What the PFedBA paper text says:
- for the remaining 6 PFL methods including FedRep, the implementation uses the hyperparameters suggested by the original research papers
- the paper text does not plainly say in one sentence that "FedRep on CIFAR-10 uses local_epochs = 20"

So the safest wording is:
- `local_epochs = 20` is a PFedBA repo and README aligned implementation setting
- it is consistent with the paper's "follow original-paper hyperparameters" statement
- but it should not be overstated as if the PFedBA paper itself directly hard-coded the number `20` in the main text for FedRep

### Why `local_epochs = 20` Likely Helps PFedBA

This setting is plausibly an important amplifier, but it should not be called the only secret.

Reason in code:
- in PFedBA FedRep, the head update phase runs for `plocal_epochs`, usually `1`
- the shared base update phase runs for `local_epochs`
- therefore `local_epochs = 20` means the shared representation is rewritten many more times per round than the personalized head

Interpretation:
- this likely helps the optimized PFedBA trigger implant its signal into the shared base more strongly
- that makes the attack more likely to survive personalization
- but the core attack advantage is still PFedBA's adaptive trigger optimization and alignment design, not the number `20` alone

### Critical CIFAR-10 Implementation Caveat

One important distinction must be kept explicit:
- the PFedBA paper defines PFedBA as a two-stage attack: trigger optimization by gradient/loss alignment, then poisoned local training
- in the authoritative GitHub-source subtree under `PFedBA/`, the visible trigger-optimization path inside the training loop is only entered for `Mnist` and `FashionMnist`
- in the visible CIFAR-10 code path of `PFedBA/FLAlgorithms/servers/serveravg.py`, there is no matching trigger-optimization branch before poisoned local training starts

Practical consequence:
- the recovered CIFAR-10 `PFedBA` logs from this GitHub-source branch should not be over-described as a fully paper-faithful adaptive-trigger implementation unless this gap is closed or another CIFAR-specific trigger-optimization path is found
- what is still solid is that this historical CIFAR-10 regime can strongly break FedRep and FedCLCM under the current code path
- but the mechanism in this branch may be closer to "paper-style regime + poisoned local training + FedRep/FedCLCM structure effects" than to the full two-stage PFedBA described in the paper

### Why The Clean Accuracy Is Lower Than Static Stage3

The lower clean accuracy in this PFedBA FedRep run should not be simplistically read as "FedRep tuning failed again".

The main reason is that this is a different regime:
- static stage3 used the clean-workspace common regime with `100` clients, `join_ratio = 0.1`, `local_epochs = 1`, `600` rounds, and static triggers
- the recovered PFedBA FedRep run used the PFedBA GitHub-source regime with `100` total clients, `10` selected users per round, `20` local epochs, `150` global rounds, and `attack_start = 30`

So the clean-accuracy gap is largely a regime difference plus stronger adaptive attack pressure.

Current practical interpretation:
- yes, hyperparameters and protocol differences are an important reason the clean acc is lower than the static `~0.79` line
- but no, the result should not be dismissed as "just a bad run"
- the critical locked fact is that under a recent paper-style PFedBA setup, FedRep still keeps non-collapsed clean accuracy while ASR becomes very high

### Relation To The PFedBA Paper Settings

The recovered Apr-16 FedRep PFedBA run is close to the PFedBA paper-style CIFAR-10 system settings at the system-regime level:
- total clients `100`
- selected clients per round `10`
- local learning rate `0.1`
- batch size `64`
- total global iterations `150`

But this should not be overstated into a mechanism claim:
- system-level hyperparameters are close
- the visible CIFAR-10 code path in the authoritative PFedBA GitHub-source subtree still needs to be distinguished from the full paper-level two-stage narrative

This is consistent with the appendix settings recorded in:
- `/home/huangtu/PFL_clean_workspace/root_static/docs/papers/usenixsecurity24-lyu.txt`

## Papers Already Identified As Most Useful
Most useful, in order:
1. `BDPFL`
2. `PFL-ALP`
3. `Bad-PFL`
4. `FLIGHT`

Supporting references:
- `usenixsecurity24-lyu.pdf`
- the user's `AdvPurge.pdf`
- the user's `main.pdf`

## Main Takeaways From The Literature

### 1. BDPFL
Why it matters:
- directly relevant to personalized backdoor defense
- explicitly useful against PFedBA-style threats
- shifts attention from server-side filtering toward representation purification

Most useful ideas:
- layer-wise mutual distillation
- less trust in deep-layer personalized features
- explanation / Grad-CAM style alignment, not just logits

Interpretation for this project:
- PFedBA is not impossible to defend
- but the defense likely needs feature-level or client-side purification

### 2. PFL-ALP
Why it matters:
- very aligned with the current suspicion that similarity-based detection is insufficient

Most useful ideas:
- clustering should mainly mitigate heterogeneity, not serve as the final malicious detector
- local purification is a major defense component
- adaptive attacks can align with benign references and bypass naive cosine-based logic

Interpretation for this project:
- continuing to add more similarity thresholds is unlikely to fundamentally solve PFedBA

### 3. Bad-PFL
Why it matters:
- explains why some static triggers become weak in PFL
- personalized local training can forget or weaken fixed triggers
- dynamic or more natural triggers can survive better

Interpretation for this project:
- if static attacks looked too weak before, that was not necessarily proof of defense strength
- but after the recent static implementation corrections, static attacks are now clearly meaningful again

### 4. FLIGHT
Why it matters:
- provides server-side scoring and clustering ideas

Why it is not the current main line:
- still mainly lives in the similarity/filtering family
- more suitable as a patch than as the main conceptual escape from PFedBA-style stress

## Methodological Judgment Already Reached
The most important conceptual conclusion from prior reading was:

PFedBA and related adaptive attacks disproportionately hurt defenses that depend mainly on similarity-based server-side filtering.

That judgment is consistent with:
- project experiments
- the user's intuition
- paper evidence

## What This Means For FedCLCM
If FedCLCM is to be improved later, the more promising direction is:
- cluster-aware representative modeling
- client-side purification
- layer-wise distillation
- feature or attention alignment

Not:
- more trim variants
- more cosine thresholds
- more server-side anomaly heuristics alone

## But Current Time Constraint Matters
Even if those method ideas are valid, thesis rescue should not immediately branch into a large redesign now.

The immediate sequence remains:
1. finish static-baseline consolidation
2. decide whether FedCLCM currently adds real value
3. only then consider the smallest defensible modification
