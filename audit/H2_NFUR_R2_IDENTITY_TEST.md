# H2 NFUR-R2 identity-off test

{
  "checkpoint": "/workspace/h2_safe_anchor_e20_medical_selected/adapter_10.pth",
  "checkpoint_sha256": "64b72dc3d1155285c826781bee4c5970bd45218e95b21675fd19d9a6b2ab54a7",
  "diffs": {
    "det_tokens_max_abs": 0.0,
    "seg_tokens_max_abs": 0.0,
    "test_maps_max_abs": 0.0,
    "train_logits_max_abs": 0.0
  },
  "finite": true,
  "identity_off_parity": "PASS",
  "paired_file_names": [
    "fryum/Data/Images/Anomaly/009.JPG",
    "candle/Data/Images/Anomaly/004.JPG",
    "fryum/Data/Images/Normal/389.JPG",
    "pcb3/Data/Images/Anomaly/062.JPG",
    "pcb2/Data/Images/Anomaly/062.JPG",
    "pcb3/Data/Images/Anomaly/024.JPG"
  ],
  "protocol_id": "H2_NFUR_R2_E20_MEDICAL_SELECTED",
  "strict_tolerance": 1e-06
}
