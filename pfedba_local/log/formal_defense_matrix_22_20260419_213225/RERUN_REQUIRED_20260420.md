# Formal Defense Matrix 22: Rerun Requirements (2026-04-20)

## Clean Rerun Decision
- All `22` tags must be rerun after the evaluation-path fix applied on `2026-04-20`.

## Why Full Rerun Is Required
- `PFL-ALP` family one-step evaluation was using the round-start global model state instead of the saved personalized warm-start before local fine-tune.
- This directly affects:
  - `E02`, `E04`, `E08`, `E10`, `E13`, `E15`, `E19`, `E21`
- Final post-training one-step evaluation in `main.py` did not redistribute the final global model to users before evaluation.
- That final-eval sync issue affects the final one-step metrics of every method, so a strict apples-to-apples comparison now requires rerunning the entire matrix.

## Operational Consequence
- Previously completed tags are no longer treated as final-valid comparison results.
- The interrupted `2026-04-20 11:41` rerun attempt was stopped intentionally before patching, so any partial progress from that launch is also discarded.

## Required Rerun Set
- `E01`, `E02`, `E03`, `E04`, `E05`, `E06`, `E07`, `E08`, `E09`, `E10`, `E11`
- `E12`, `E13`, `E14`, `E15`, `E16`, `E17`, `E18`, `E19`, `E20`, `E21`, `E22`
