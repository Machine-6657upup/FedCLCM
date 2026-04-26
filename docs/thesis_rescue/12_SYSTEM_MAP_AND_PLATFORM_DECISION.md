# System Map And Platform Decision

## Why This File Exists
The project now spans multiple partially overlapping code lines. The main confusion is no longer just "which result is better", but:
- which repository is the real source for a given mechanism
- which logs belong to which code line
- whether dynamic attacks should be forced back into `main.py`

This file records only what was directly checked in code and log directories.

## Directly Checked Code Lines

### 1. Clean static-mainline workspace
Main path:
- `/home/huangtu/PFL_clean_workspace/root_static`

Direct checks:
- `src/main.py` exposes `--algorithm FedRep/FedRT/FedCLCM`
- `src/main.py` also exposes `--attack badpfl/pfedba`
- but there is no `src/attacks/` directory in this workspace
- therefore `root_static` currently supports static backdoor experiments as the main runnable line, not a complete dynamic-attack-native line

Static attack generation really lives here:
- `src/dataset/utils/generate_Cifar10_badnet.py`
- `src/dataset/utils/generate_Cifar10_blend.py`
- `src/dataset/utils/generate_Cifar10_sig.py`

Key defense implementations really live here:
- `src/User/serverRepTrim.py`
- `src/User/clientrepclidef.py`
- `src/User/serverCLCM.py`
- `src/User/clientCLCM.py`

### 2. Old main framework
Main path:
- `/home/huangtu/PFL_Backdoor_Defense`

Direct checks:
- old `main.py` and clean `src/main.py` are byte-identical
- old `User/serverRepTrim.py` and clean `src/User/serverRepTrim.py` are byte-identical
- old `User/clientrepclidef.py` and clean `src/User/clientrepclidef.py` are byte-identical
- old `User/serverCLCM.py` and clean `src/User/serverCLCM.py` are byte-identical
- old `User/clientCLCM.py` and clean `src/User/clientCLCM.py` are byte-identical

Working meaning:
- the clean workspace is not a different defense logic branch
- it is a cleaned execution and documentation branch built on the same core FedRT / FedCLCM code line
- the main difference is experiment hygiene, dataset repair, run management, and documentation

### 3. Official PFedBA subrepository
Main path:
- `/home/huangtu/PFL_Backdoor_Defense/PFedBA`

Direct checks:
- git remote points to `https://github.com/xtLyu/PFedBA.git`
- `main.py` parser default is `local_epochs = 20`
- `main.py` parser default is `plocal_epochs = 1`
- supported algorithms explicitly include `FedAvg`, `FedProx`, `FedRep`, `FedCLCM`
- `FLAlgorithms/users/userrep.py` implements FedRep as split `head` plus shared `base`
- `FLAlgorithms/users/userrep.py` updates head for `plocal_epochs`, then base for `local_epochs`
- `FLAlgorithms/users/userclcm.py` is a ported FedCLCM client on top of the PFedBA stack

Working meaning:
- PFedBA native evaluation should be treated as a separate authoritative attack platform
- if we want faithful PFedBA comparisons, this repository is the correct place to extend algorithms

### 4. Official Bad-PFL subrepository
Main path:
- `/home/huangtu/PFL_Backdoor_Defense/Bad-PFL`

Direct checks:
- git remote points to `https://github.com/fmy266/Bad-PFL.git`
- `main.py` parser default is `total_round = 300`
- `main.py` parser default is `client_local_step = 15`
- `main.py` parser default is `pfl = fedbn`
- `pfl.py` currently exposes a `use_fedbn(server)` path
- `fba.py` contains the dynamic trigger generator plus PGD-based poisoning logic

Working meaning:
- Bad-PFL is also its own native attack platform
- out of the box, this code line is centered on `FedBN`, not a broad PFL benchmark suite
- if we want Bad-PFL on FedRep / FedRT / FedCLCM, that extension should be done inside this native repository, not by forcing Bad-PFL logic into `root_static`

## Directly Checked Log Families By Date

### 2026-03-25
Old thesis basic static line under:
- `/home/huangtu/PFL_Backdoor_Defense/thesis_log/main_results_basic`

This is the older "main thesis baseline" batch.

### 2026-04-08
Static attack generalization under:
- `/home/huangtu/PFL_Backdoor_Defense/thesis_log/attack_generalization_static`

This is the older static cross-attack comparison batch.

### 2026-04-09 to 2026-04-14
FedCLCM ablation and sensitivity families under:
- `thesis_log/module_ablation`
- `thesis_log/client_sensitivity`
- `thesis_log/heterogeneity_study`
- `thesis_log/efficiency_overhead`
- `thesis_log/badnet_fedclcm_training_hp`

These belong to the old main-framework thesis exploration line.

### 2026-04-16
Recent PFedBA comparison anchors under:
- `/home/huangtu/PFL_Backdoor_Defense/logs/pfedba_cifar10_quad_20260416_165930`
- `/home/huangtu/PFL_Backdoor_Defense/PFedBA/results/Cifar10_Apr.16_16.59.33_FedRep_attackall_none_5_1`
- `/home/huangtu/PFL_Backdoor_Defense/PFedBA/results/Cifar10_Apr.16_16.59.33_FedCLCM_attackall_none_5_1`

This is the strongest recent dynamic-attack anchor currently recovered for CIFAR-10 PFedBA.

### 2026-04-17 to 2026-04-18
Clean static rescue line under:
- `/home/huangtu/PFL_clean_workspace/root_static/runs/20260417_184038_stage3_static_paper_common`

This is the current clean thesis-rescue baseline line and should remain the static source of truth.

## Trust Boundary That Must Be Kept
PFedBA mechanism claims:
- trust `/home/huangtu/PFL_Backdoor_Defense/PFedBA`
- do not trust PFedBA-like code outside that subfolder as mechanism evidence

Bad-PFL mechanism claims:
- trust `/home/huangtu/PFL_Backdoor_Defense/Bad-PFL`
- do not treat hand-written or later-generated imitation code elsewhere as equally authoritative

Static thesis-rescue claims:
- trust `/home/huangtu/PFL_clean_workspace/root_static`

## Platform Decision
Current judgment based on direct code inspection:

1. A fully unified platform is not required right now.
2. Static attacks should stay in `root_static`.
3. PFedBA should stay in the official `PFedBA/` repository for faithful dynamic evaluation.
4. Bad-PFL should stay in the official `Bad-PFL/` repository for faithful dynamic evaluation.
5. What must be unified is the evaluation protocol and thesis tables, not necessarily the execution codebase.

## Practical Engineering Consequence
The most practical structure is:

### Static line
Run and document in:
- `/home/huangtu/PFL_clean_workspace/root_static`

### PFedBA line
Add or maintain FL / PFL baselines in:
- `/home/huangtu/PFL_Backdoor_Defense/PFedBA`

### Bad-PFL line
Add or maintain FL / PFL baselines in:
- `/home/huangtu/PFL_Backdoor_Defense/Bad-PFL`

## What This Means For The Thesis Route
This supports the user's current instinct:
- do not force PFedBA and Bad-PFL back into `main.py` if that damages reproduction fidelity
- instead, keep attack-native code where it is strongest
- then normalize the final comparison in thesis tables, metrics, and narrative

In other words:
- code platform can remain split
- thesis evidence must remain aligned

## Current Bottom-Line Judgment
At this stage, the safest graduation-oriented route is:
- keep `root_static` as the static benchmark source of truth
- treat `PFedBA/` as the authoritative dynamic PFedBA platform
- treat `Bad-PFL/` as the authoritative dynamic Bad-PFL platform
- later port the final defense to those native dynamic platforms in the smallest possible way
- avoid another large cross-platform rewrite before the evidence chain is stable
