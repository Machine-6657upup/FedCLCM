# Thesis Rescue Master Index

## Purpose
This document set is the handoff package for the thesis rescue workflow. It is meant to replace fragile chat-context memory with explicit, reviewable files inside the clean workspace.

The goal is not to restate every conversation detail. The goal is to preserve the useful decisions, confirmed facts, current experiment status, and immediate next actions in a way that a fresh Codex session can reliably continue.

## Core Rule
Do not rely on hidden or previous chat context.

Only trust:
- these handoff documents
- code in the repositories
- logs and result files
- experiment scripts and run directories

## Immediate Priority
The current priority is not to invent a large new defense.

The immediate priority is to finish stabilizing the static-backdoor story:
1. show FedRep can achieve credible clean accuracy under a unified common regime
2. show BadNet / SIG / Blend are not so weak that FedRep is impossible to attack
3. compare FedAvg / FedRep / FedCLCM under the same static settings
4. produce a clean, defendable thesis narrative from those results

PFedBA is important, but it is temporarily secondary until the static line is fully stable.

## Reading Order
Read these files in order:
1. `01_PROJECT_GOAL_AND_PAIN_POINTS.md`
2. `02_HISTORICAL_LOG_RECOVERY.md`
3. `03_CODEBASE_AND_DATASET_AUDIT.md`
4. `04_STAGE3_STATIC_EXPERIMENTS.md`
5. `05_PFEDBA_AND_LITERATURE.md`
6. `06_CONFIRMED_DECISIONS.md`
7. `07_NEXT_ACTIONS.md`
8. `08_REMOTE_OPERATING_RULES.md`
9. `12_SYSTEM_MAP_AND_PLATFORM_DECISION.md`
10. `13_PFEDBA_DEFENSE_ROUTE_AND_HEAD_EPOCH.md`
11. `14_STATIC_FEDCLCM_RESCUE_ANALYSIS.md`

## Current High-Confidence Conclusions
- The old claim "FedRep was clearly broken by static backdoors before" is real and backed by historical logs.
- The recent confusion did not come from a single cause. It came from mixed code paths, mixed datasets, inconsistent hyperparameters, weak SIG/Blend implementations, and messy result directories.
- After moving into the clean workspace and fixing static attack generation, the static story is now viable again.
- Stronger BadNet and corrected SIG are definitely not "too weak to matter".
- FedRep under the unified common regime is no longer stuck in the low-0.7x region. It recovered to a credible baseline.
- FedCLCM currently has a major interpretation problem: low ASR often comes together with collapsed clean accuracy, so it cannot be counted as a clean defense win.
- Old FedCLCM/CLCM CIFAR-10 low-ASR results are real, but they were obtained mainly under `nclient=40`, `adv5`, `ResNetP`, and old optimizer/PGD settings. No old CIFAR-10 `nclient=100` FedCLCM success log has been found yet.

## Key Locations
Remote clean workspace:
- `/home/huangtu/PFL_clean_workspace/root_static`

Local clean workspace:
- `C:\Users\12709\remote_edit\root_static`

Historical mixed repository:
- `/home/huangtu/PFL_Backdoor_Defense`

Current static stage3 run:
- `/home/huangtu/PFL_clean_workspace/root_static/runs/20260417_184038_stage3_static_paper_common`

Downloaded local copies of stage3 logs:
- `C:\Users\12709\remote_edit\root_static\downloads\20260417_stage3_logs`

## Papers That Should Exist On The Server
These are the most useful papers to keep on the server for the next Codex session:
- `BDPFL`
- `PFL-ALP`
- `Bad-PFL`
- `FLIGHT`
- `usenixsecurity24-lyu.pdf`
- `AdvPurge.pdf`
- `main.pdf`

Recommended remote directory:
- `/home/huangtu/PFL_clean_workspace/root_static/docs/papers`

## Required First Response From A Fresh Remote Codex Session
After reading the handoff docs, the assistant should first output:
1. the current overall objective
2. the historical facts that are already locked
3. the top 3 next actions

Before that is done, it should not propose large new methods or diffuse into unrelated branches.
