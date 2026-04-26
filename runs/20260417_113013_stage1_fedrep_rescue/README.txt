Stage 1: FedRep baseline rescue
Timestamp: 20260417_113013

Purpose:
- recover a strong and reproducible FedRep baseline
- focus on the historical 40-client badnet line
- avoid mixing this stage with the current 100-client thesis mainline

Dataset:
- Cifar10_dir0.5_bdoor0.2_nclient_40_badnet_adv5

Fixed settings:
- num_clients = 40
- num_adv_clients = 5
- local_epochs = 1
- plocal_epochs = 1
- batch_size = 64
- global_rounds = 800

Sweep factors:
- model: ResNet18 vs ResNetP
- join_ratio: 1.0 vs 0.1
- learning-rate family: narrow manual rescue sweep
