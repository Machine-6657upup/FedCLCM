# Codebase And Dataset Audit

## Repository Situation
The original project repository evolved over a long time and contains mixed history:
- original experiments
- modified algorithms
- multiple result folders
- ad hoc scripts
- Cursor-generated scripts
- historical leftover dataset-generation logic

This is why a clean workspace was created.

## Main Paths
Remote clean workspace:
- `/home/huangtu/PFL_clean_workspace/root_static`

Local clean workspace:
- `C:\Users\12709\remote_edit\root_static`

Historical mixed repository:
- `/home/huangtu/PFL_Backdoor_Defense`

## PFedBA Source Authority Rule
For PFedBA-specific code interpretation, the authoritative source is narrower than the general historical repository:
- trusted PFedBA code source: `/home/huangtu/PFL_Backdoor_Defense/PFedBA`
- this path corresponds to the GitHub-source subtree and related logic
- PFedBA-like code outside that subfolder in the historical repository should not be treated as authoritative evidence

Working implication:
- use the historical repository broadly for old logs and comparison
- but use the `PFedBA/` GitHub-source subtree only when making PFedBA mechanism claims

## Why Dataset Generation Was Treated As A Critical Risk
The user explicitly warned that dataset generation had historical technical debt:
- based on incremental modifications from `Backdoor&PFL tool\PFLlib\dataset`
- some hardcoded logic existed
- some logic quality was poor
- if dataset generation was wrong, all later experiments could be misleading

This warning was correct and had to be treated seriously.

## Important Audit Conclusion
The dataset-generation path was indeed messy enough to justify suspicion.

However, after fixing the static attack generation in the clean workspace, the new runs showed:
- BadNet can be strong
- SIG can be very strong
- Blend can be noticeably stronger than before

So the current working conclusion is:
- dataset generation was a real risk and needed repair
- but it is not the only root cause of the earlier chaos
- weak SIG/Blend implementation and mixed hyperparameters were also major causes

## Static Attack Generation Changes Already Made

### File: `scripts/generate_dataset_via_existing.py`
Added generation arguments:
- `--blend-alpha`
- `--sig-delta`
- `--sig-f`
- `--sig-label-mode`

These are now written into the generation fingerprint for traceability.

### File: `src/dataset/utils/generate_Cifar10_blend.py`
Changes:
- removed debug image dumping
- parameterized `blend_alpha`
- current working value in stage3: `blend_alpha = 0.2`
- clip outputs

### File: `src/dataset/utils/generate_Cifar10_sig.py`
Changes:
- removed hardcoded `rawdata_path`
- parameterized:
  - `sig_delta`
  - `sig_f`
  - `sig_label_mode`
- current working default:
  - `sig_delta = 30/255`
  - `sig_f = 6`
  - `sig_label_mode = dirty`
- fixed normalized-space scaling by mapping the raw perturbation correctly into normalized image space
- clip outputs

## Why The SIG Fix Was Important
One major source of confusion was that SIG looked too weak in recent experiments.

After the correction:
- SIG became substantially stronger
- FedAvg under SIG became clearly compromised
- FedRep under SIG showed nontrivial ASR

This strongly supports the claim that the earlier SIG weakness was at least partly an implementation-strength problem, not a proof that personalized models automatically neutralize SIG.

## Codebase Working Principle Going Forward
1. Use the clean workspace for new mainline experiments.
2. Use the historical repository only for old-log lookup and old-code comparison.
3. Keep attack-generation parameters explicit and fingerprinted.
4. Do not tolerate hidden hardcoded dataset paths in the mainline.
