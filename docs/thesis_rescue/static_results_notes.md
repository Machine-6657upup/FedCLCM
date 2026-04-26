# Static Results Notes

## Completion Check
- eight of the nine stage3 jobs fully finished to round `600` and ended with `All done!`
- the only incomplete job is `s3_clcm_sig`
- `s3_clcm_sig` reached round header `65`, but the last complete metric block in the raw log is round `64`

## Convergence Judgment

FedAvg:
- all three attacks show late-stage convergence
- best rounds are `600`, `598`, and `592`
- final accuracy remains close to the best point
- ASR is high under all three attacks, especially BadNet and SIG

FedRep:
- all three attacks also converge late
- best rounds are `570`, `498`, and `523`
- final clean accuracy stays near `0.79`
- final accuracy is very close to the best point in all three runs
- the attacks remain meaningful against FedRep:
  - BadNet final ASR `0.5755`
  - SIG final ASR `0.3393`
  - Blend final ASR `0.1794`

FedCLCM:
- the pattern is different and much worse
- `s3_clcm_badnet` peaks at round `41` with best acc `0.5091`, then ends at `0.4033`
- `s3_clcm_blend` peaks at round `34` with best acc `0.4632`, then ends at `0.4037`
- `s3_clcm_sig` peaks at round `38` with best acc `0.4720`, then falls to `0.3833` by its last complete round `64`
- these are collapse signatures, not stable convergence signatures

## Curve-Based Interpretation
- FedAvg and FedRep curves rise through mid and late training and then plateau near their best rounds
- FedCLCM curves peak early and spend the remainder of training far below the peak clean accuracy
- low FedCLCM ASR cannot be credited as a defense success because it arrives together with damaged clean accuracy

## Static Thesis Conclusions
1. the corrected static attacks are strong enough to sustain a credible thesis comparison
2. FedRep is not untouchable under the unified common regime
3. FedRep now has a credible clean-accuracy baseline around `0.79`
4. FedCLCM does not currently provide a clean, convincing static-defense win over FedRep

## What This Means For The Rescue Workflow
- the static line is now stable enough to support writing tables and thesis narrative
- do not pivot back to PFedBA until the static comparison line is treated as the settled baseline
- if later cleanup is needed, the only unresolved stage3 run is `s3_clcm_sig`
