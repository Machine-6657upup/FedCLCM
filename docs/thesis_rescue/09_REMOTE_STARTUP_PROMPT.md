# Remote Startup Prompt

Use the following as the first prompt in a fresh remote Codex session:

```text
First read `docs/thesis_rescue/00_MASTER_INDEX.md` through `docs/thesis_rescue/08_REMOTE_OPERATING_RULES.md`.

After reading them, do not propose new broad methods yet. First output:
1. the current overall thesis rescue goal
2. the historical facts that are already locked
3. the current top 3 next actions

Then continue the rescue workflow in this order:
1. check whether the stage3 static runs fully finished
2. extract best_acc / best_round / final_acc / final_asr for every stage3 task
3. judge convergence vs collapse from the logs and curves
4. produce a clean static comparison table for FedAvg / FedRep / FedCLCM under BadNet / Blend / SIG
5. write the conclusions back into docs, rather than leaving them only in chat output

Important constraints:
- do not rely on hidden previous chat memory
- use `/home/huangtu/PFL_clean_workspace/root_static` as the main workspace
- use `/home/huangtu/PFL_Backdoor_Defense` only for historical comparison
- do not treat low ASR as a defense win if clean accuracy also collapses
- do not pivot back to PFedBA until the static line is fully stabilized
```
