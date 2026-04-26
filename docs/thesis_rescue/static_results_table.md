# Static Results Table

## Source
- run directory: `/home/huangtu/PFL_clean_workspace/root_static/runs/20260417_184038_stage3_static_paper_common`
- summary file: `/home/huangtu/PFL_clean_workspace/root_static/runs/20260417_184038_stage3_static_paper_common/metrics_summary.csv`
- curves: `/home/huangtu/PFL_clean_workspace/root_static/runs/20260417_184038_stage3_static_paper_common/curves/`
- special note: `s3_clcm_sig` is incomplete, so its entry uses the last complete metric block from the raw log at round `64`

## Per-Task Accounting

| Task | Algorithm | Attack | Status | Best Acc | Best Round | Final Acc | Final ASR |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `s3_avg_badnet` | FedAvg | BadNet | done | 0.5130 | 600 | 0.5130 | 0.9345 |
| `s3_avg_blend` | FedAvg | Blend | done | 0.5267 | 598 | 0.5252 | 0.7112 |
| `s3_avg_sig` | FedAvg | SIG | done | 0.5188 | 592 | 0.4968 | 0.9934 |
| `s3_rep_badnet` | FedRep | BadNet | done | 0.7887 | 570 | 0.7870 | 0.5755 |
| `s3_rep_blend` | FedRep | Blend | done | 0.7954 | 498 | 0.7921 | 0.1794 |
| `s3_rep_sig` | FedRep | SIG | done | 0.7948 | 523 | 0.7906 | 0.3393 |
| `s3_clcm_badnet` | FedCLCM | BadNet | done | 0.5091 | 41 | 0.4033 | 0.0462 |
| `s3_clcm_blend` | FedCLCM | Blend | done | 0.4632 | 34 | 0.4037 | 0.0650 |
| `s3_clcm_sig` | FedCLCM | SIG | partial to round 64 | 0.4720 | 38 | 0.3833 | 0.0336 |

## Static Comparison Table

| Attack | FedAvg | FedRep | FedCLCM |
| --- | --- | --- | --- |
| BadNet | best `0.5130 @ 600`; final `0.5130`; ASR `0.9345` | best `0.7887 @ 570`; final `0.7870`; ASR `0.5755` | best `0.5091 @ 41`; final `0.4033`; ASR `0.0462` |
| Blend | best `0.5267 @ 598`; final `0.5252`; ASR `0.7112` | best `0.7954 @ 498`; final `0.7921`; ASR `0.1794` | best `0.4632 @ 34`; final `0.4037`; ASR `0.0650` |
| SIG | best `0.5188 @ 592`; final `0.4968`; ASR `0.9934` | best `0.7948 @ 523`; final `0.7906`; ASR `0.3393` | partial: best `0.4720 @ 38`; last complete `0.3833 @ 64`; ASR `0.0336` |

## Guardrail For Interpretation
- low ASR is not counted as a defense win when clean accuracy collapses
- this guardrail is decisive for the current FedCLCM interpretation
