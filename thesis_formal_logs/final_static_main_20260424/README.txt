Final static main folder for the thesis.

Purpose:
- preserve only the static logs that are safe to reuse for the thesis main table
- keep reference-only historical logs separated from thesis-main logs
- rerun only the genuinely missing or mismatched static experiments

Current static main setting:
- dataset family = Cifar10_dir0.5_bdoor0.2_nclient_100_{badnet,blend,sig}_adv10
- model = ResNet18
- num_clients = 100
- join_ratio = 0.1
- local_learning_rate = 0.1
- lr_head = 0.1 for FedRep and FedCLCM
- local_epochs = 1
- plocal_epochs = 1 for FedRep and FedCLCM
- num_adv_clients = 10
- global_rounds = 600

Reused historical batch:
- source run = runs/20260417_184038_stage3_static_paper_common
- these reused logs were produced under the same dataset family and core training setting
- that batch used eval_gap = 1
- only FedAvg and FedRep are currently reusable from that batch for the thesis main table

Reference-only historical logs:
- the old static FedCLCM logs were moved to reference_only_lr_head_mismatch/
- they are not thesis-main reusable because the source run used lr_head = 0.005
- the thesis target setting for FedCLCM is lr_head = 0.1, so those three tasks must be rerun

New missing runs:
- FedTrimmed remains missing for badnet, blend, and sig
- FedCLCM must be rerun for badnet, blend, and sig under lr_head = 0.1
- new runs can use eval_gap = 10
- evaluation frequency does not change the training trajectory here; it only reduces evaluation overhead
