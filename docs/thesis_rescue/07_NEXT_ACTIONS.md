# Next Actions

## Immediate Goal
Finish the static attack consolidation workflow and produce thesis-usable artifacts.

## Priority Order

### Priority 1: complete the stage3 result accounting
Check and record for each stage3 job:
- whether the job fully finished
- best accuracy
- best-accuracy round
- final accuracy
- final ASR
- whether the curve converged
- whether the model collapsed

The unresolved or partially synced items are:
- `s3_rep_badnet`
- `s3_rep_blend`
- `s3_clcm_blend`
- `s3_clcm_sig`

### Priority 2: produce a static comparison table
The comparison should be structured as:
- rows: `BadNet`, `Blend`, `SIG`
- columns: `FedAvg`, `FedRep`, `FedCLCM`
- each cell should report at least:
  - best acc
  - final acc
  - final ASR

### Priority 3: inspect curves
Do not stop at scalar metrics.

For each completed run, inspect:
- clean accuracy curve
- ASR curve
- signs of convergence
- signs of overfitting
- signs of collapse

### Priority 4: write the result interpretation into docs
Create thesis-ready internal notes that say:
- what static experiments now prove
- what they do not yet prove
- why FedCLCM low ASR currently cannot be trusted as a full win

## After Static Consolidation
Only after the above is complete:
1. revisit PFedBA
2. first recover and tabulate the recent PFedBA log anchors before launching new reruns
3. use the recovered `2026-04-16` FedRep / FedCLCM CIFAR-10 PFedBA results as the comparison starting point
4. decide whether FedCLCM needs a minimal modification
5. if so, strongly prefer the smallest meaningful modification rather than a broad redesign

## What Not To Do Next
- do not reopen too many historical branches
- do not immediately launch a large batch of PFedBA runs
- do not propose a big new defense architecture before static results are tabulated and written down
- do not treat one-off lucky numbers as the new main story

## Deliverables That Should Exist Soon
- `docs/thesis_rescue/static_results_table.md`
- `docs/thesis_rescue/static_results_notes.md`
- plots under `docs/plots/` or another stable documented path
- a short thesis-facing summary paragraph that can later be inserted into the dissertation
