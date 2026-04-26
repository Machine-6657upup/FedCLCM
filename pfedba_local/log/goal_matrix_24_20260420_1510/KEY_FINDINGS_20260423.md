# Key Findings

Date: 2026-04-23
Run directory: `pfedba_local/log/goal_matrix_24_20260420_1510`

## Most Important Conclusion

`FedCLCM` is currently the strongest method in the PFedBA-on-FedRep matrix.

This is not a weak or marginal result. It is the cleanest positive conclusion in the whole matrix because:

1. It improves the standard `FedRep` personalized evaluation metric (`k local SGD`), not only a method-specific saved-model inference path.
2. It beats `FedRT` on the same matrix.
3. It has much better acc-ASR tradeoff than the current `BDPFL-lite` transfer modules.

## Evidence

Main comparison metric:

- `Average Personal Accurancy (k local SGD)`
- `Average Personal ATTACK ALL ASR (k local SGD)`

These are the directly comparable `FedRep`-style personalized metrics.

### LE10 / GI400

Baseline:

- `E01 FedRep` from `formal_defense_matrix_22_20260419_213225`
- personal acc / ASR = `0.8000 / 0.5437`

FedCLCM:

- `E07`
- personal acc / ASR = `0.7777 / 0.2750`
- delta vs baseline = `-0.0223 acc`, `-0.2687 ASR`

FedRT:

- `E06`
- personal acc / ASR = `0.7724 / 0.4642`

Interpretation:

- `FedCLCM` keeps almost all clean accuracy compared with baseline.
- `FedCLCM` cuts ASR much more aggressively than `FedRT`.
- On `LE=10`, `FedCLCM` is the best overall tradeoff among all finished methods in the matrix.

### LE1 / GI1000

Baseline:

- `E12 FedRep`
- personal acc / ASR = `0.7367 / 0.3891`

FedCLCM:

- `E18`
- personal acc / ASR = `0.7754 / 0.1786`
- delta vs baseline = `+0.0386 acc`, `-0.2105 ASR`

FedRT:

- `E17`
- personal acc / ASR = `0.6844 / 0.2318`

Interpretation:

- `FedCLCM` is especially strong at `LE=1`.
- It is the only method in this matrix that simultaneously improves personalized acc and strongly suppresses ASR.
- This is the strongest single result in the current experiment matrix.

## FedCLCM Hyperparameters To Keep

The current effective PFedBA/FedRep-aligned `FedCLCM` setting is:

- `rt_beta=0.20`
- `lambda_cl=0.20`
- `mask_tau=12.0`
- `mask_alpha=0.70`
- `adv_eps=0`
- `adv_num_iter=0`
- `enable_channel_mask=1`
- `cosine_gate=0`

Matrix runs:

- `E07`: `local_epochs=10`, `num_global_iters=400`
- `E18`: `local_epochs=1`, `num_global_iters=1000`

Shared matrix settings:

- `dataset=Cifar10`
- `model=resnet`
- `learning_rate=0.1`
- `lr_head=0.1`
- `plocal_epochs=1`
- `numusers=10`
- `batch_size=64`
- `attack_start=30`
- `attack_method=attackall`
- `poisoning_per_batch=1`
- `malclient=10`

## Practical Takeaway

For the current short-term project goal:

- `FedCLCM` should be treated as a priority strong baseline / comparison target.
- `LE=1` + `FedCLCM` is currently the most promising point.
- `FedRT` remains a useful comparison method, but it is not the best result anymore.
- `PFLALP-lite` should not be prioritized over `FedCLCM`.
- `BDPFL-lite` still has research value, but only as a defense-paper transfer direction with a much harder acc-ASR tradeoff.

## Exact Log References

- `E07`: `pfedba_local/log/goal_matrix_24_20260420_1510/E07.log`
- `E18`: `pfedba_local/log/goal_matrix_24_20260420_1510/E18.log`
- `E06`: `pfedba_local/log/goal_matrix_24_20260420_1510/E06.log`
- `E17`: `pfedba_local/log/goal_matrix_24_20260420_1510/E17.log`
- `E12`: `pfedba_local/log/goal_matrix_24_20260420_1510/E12.log`
- `E01`: `pfedba_local/log/formal_defense_matrix_22_20260419_213225/E01.log`
