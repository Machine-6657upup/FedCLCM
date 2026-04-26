Paper-aligned full-method launcher

This launcher is separated from the FedRep lite/baseline matrix on purpose.
- PFLALP full:
  round=100, benign local epoch=1, attacker local epoch=6, lr=0.1, attacker lr=0.05
  malicious pool=30, fixed malicious per round=3
- BDPFL full:
  round=1000, local epoch=20, lr=0.1, lr decay=0.99 every 10 rounds
  attackers=3, fixed malicious per round=3

Repo-specific note:
- POISONING_PER_BATCH=1 is a PFedBA attack knob and does not have a direct 1:1 paper counterpart.
