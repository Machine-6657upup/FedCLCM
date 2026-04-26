# Historical Log Recovery

## Why Historical Recovery Mattered
The user explicitly remembered that, during the earlier small-paper period, FedRep had been clearly broken by static attacks while AdvPurge remained relatively stable.

That memory was critical because if it were false, the thesis static narrative would be severely weakened.

The historical-log search established that the memory was real.

## Locked Historical Strong Log
The key historical log that confirms this is:
- `/home/huangtu/PFL_Backdoor_Defense/new_result/cifar/FedRep_Cifar10_dir0.5_bdoor0.2_nclient_40_badnet_adv5.log`

## Locked Historical Configuration
This strong historical run corresponds to approximately:
- `num_clients = 40`
- `join_ratio = 0.25`
- `model = ResNetP`
- `local_learning_rate = 0.1`
- `lr_head = 0.05`
- `local_epochs = 1`
- `plocal_epochs = 1`
- `global_rounds = 600`
- `num_adv_clients = 5`

## Meaning Of This Historical Fact
This historical evidence means:
1. the earlier narrative that "FedRep was broken by static backdoors" was not fabricated
2. the user's thesis direction did at one point rest on a real experimental outcome
3. the recent contradictory results must be explained by changes in setup, attack strength, code path, or data path, not by "the old result never existed"

## Important Strategic Shift
Although the old strong result is real, the team explicitly decided not to spend the thesis rescue phase on exact 88+ reproduction as the highest priority.

Reason:
- the history is already locked by logs
- the thesis can be rescued with a credible present baseline
- exact numerical nostalgia is less important than a stable and explainable current story

## The Correct Use Of Historical Logs Now
Historical logs should now be used as:
- a reality anchor
- a parameter reference
- proof that static vulnerability in FedRep is plausible

They should not dominate the entire new schedule.

## Current Practical Interpretation
The rescue phase should treat the historical log as evidence for this sentence:

"FedRep being broken by static backdoors is a real phenomenon in this project history, so the current task is not to prove possibility from scratch, but to rebuild a stable and credible version of that story under unified, cleaner settings."
