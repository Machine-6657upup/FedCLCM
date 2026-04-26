Reference-only static logs.

These three historical FedCLCM logs are kept only for inspection and comparison.

They are not safe to use in the thesis main table because:
- the historical source batch used lr_head = 0.005
- the thesis target static setting uses lr_head = 0.1

Affected tasks:
- S04_badnet_fedclcm.log
- S08_blend_fedclcm.log
- S12_sig_fedclcm.log
