Final PFedBA formal folder for the thesis.

Purpose:
- preserve PFedBA logs that are already aligned with the current thesis setting
- separate main-table reusable logs from local-epoch reference logs
- record which PFedBA experiments still need to be run

Current PFedBA thesis setting:
- dataset = Cifar10
- model = resnet
- resnet_pretrained = 0
- total_users = 100
- numusers = 10
- learning_rate = 0.1
- lr_head = 0.1
- plocal_epochs = 1
- batch_size = 64
- attack_start = 30
- attack_method = attackall
- poisoning_per_batch = 1
- per_epoch = 1
- default malicious clients = 10

Reusable main logs in this folder:
- main_single_seed_reused/P01_fedrep_le1_gi1000_E12.log
- main_single_seed_reused/P02_fedrt_le1_gi1000_E17.log
- main_single_seed_reused/P03_fedclcm_le1_gi1000_E18.log

Important caveat:
- these three main logs come from goal_matrix_24_20260420_1510 with times = 1
- they are valid single-seed aligned logs
- if the thesis main table uses 3-seed mean/std, then FedRep, FedRT, and FedCLCM still need seed completion or a unified 3-seed rerun

Reusable reference logs in this folder:
- reference_le10_reused/R01_fedrt_le10_gi400_E06.log
- reference_le10_reused/R02_fedclcm_le10_gi400_E07.log

Known missing item from the same source batch:
- E01 FedRep LE10 / GI400 is listed in the source manifest but its log file is absent from the source directory
- so the LE10 FedRep point cannot currently be reused and must be rerun if the epoch study keeps that row

Source snapshot:
- source_snapshot/source_goal_matrix_24_manifest.txt
