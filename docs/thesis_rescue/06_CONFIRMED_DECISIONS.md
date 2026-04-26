# Confirmed Decisions

This file records decisions that were already discussed and should not be re-litigated unless new hard evidence appears.

## Decision 1
Do not make exact historical `88+` reproduction the top priority.

Reason:
- the historical strong result has already been proven real by logs
- thesis rescue now depends more on a stable, current, credible baseline than on exact nostalgia reproduction

## Decision 2
Use the unified common regime as the current mainline for static experiments.

Current accepted core values:
- `num_clients = 100`
- `join_ratio = 0.1`
- `local_learning_rate = 0.1`
- `local_epochs = 1`
- `plocal_epochs = 1`
- `model = ResNet18`
- `num_adv_clients = 10`

## Decision 3
Use dirty-label SIG for now.

Reason:
- the immediate goal is to align implementation strength with mainstream practical logic
- not to branch into label-mode debates before the mainline is stable

## Decision 4
Prioritize fixing attack implementation strength before declaring the defense good.

Reason:
- earlier SIG and Blend looked suspiciously weak
- corrected implementations already changed the story substantially

## Decision 5
Do not count low ASR as success unless clean accuracy remains healthy.

This is especially important for FedCLCM, where low ASR may currently be coupled to collapse.

## Decision 6
PFedBA is important, but temporarily secondary.

Reason:
- static-baseline instability must be solved first
- otherwise PFedBA only adds more ambiguity

## Decision 7
Use the clean workspace as the only forward mainline.

Remote:
- `/home/huangtu/PFL_clean_workspace/root_static`

Local:
- `C:\Users\12709\remote_edit\root_static`

The historical repository remains useful for:
- old logs
- code comparison
- old parameter tracing

but not for new mainline experimentation.

## Decision 8
Do not let the thesis rescue branch into many new ideas before tables and curves for the static line are complete.

The project is in a deadline-driven narrowing phase, not an open-ended exploratory phase.

## Decision 9
Do not force dynamic attacks into the clean `main.py` framework if that harms reproduction fidelity.

Current practical split:
- static attacks stay in `/home/huangtu/PFL_clean_workspace/root_static`
- PFedBA stays in `/home/huangtu/PFL_Backdoor_Defense/PFedBA`
- Bad-PFL stays in `/home/huangtu/PFL_Backdoor_Defense/Bad-PFL`

What must be unified is:
- metrics
- tables
- protocol description
- thesis narrative

What does not need to be unified immediately is:
- the execution codebase for every attack family
