Stage 3: static attacks under paper-common regime
Timestamp: 20260417_184038

Purpose:
- align static experiments with a paper-common PFL regime
- validate stronger dirty-label Blend and SIG implementations
- compare FedAvg and FedRep first, then FedCLCM if a GPU is available

Common regime:
- num_clients = 100
- join_ratio = 0.1
- lr = 0.1
- lr_head = 0.1
- local_epochs = 1
- plocal_epochs = 1
- model = ResNet18
- num_adv_clients = 10
- global_rounds = 600

Attack-specific generation defaults:
- blend_alpha = 0.2
- sig_delta = 0.11764705882352941
- sig_f = 6
- sig_label_mode = dirty
