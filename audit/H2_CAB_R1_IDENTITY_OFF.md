# H2 CAB-LoRA R1 Identity-Off Audit

{
  "cab_enabled": false,
  "checkpoint": "/workspace/h2_safe_anchor_e20_medical_selected/adapter_10.pth",
  "checkpoint_sha256": "64b72dc3d1155285c826781bee4c5970bd45218e95b21675fd19d9a6b2ab54a7",
  "diffs_max_abs": {
    "final_anomaly_map": 0.0,
    "final_fused_logits": 0.0,
    "stage1_logits": 0.0,
    "stage2_logits": 0.0,
    "stage3_logits": 0.0,
    "task_loss": 0.0
  },
  "finite": true,
  "graph_shapes": {
    "final_logits": [
      1,
      2,
      518,
      518
    ],
    "final_map": [
      1,
      518,
      518
    ],
    "stage2_post_replay": [
      1369,
      1,
      1024
    ],
    "stage2_pre": [
      1369,
      1,
      1024
    ],
    "stage_logits": [
      3,
      1,
      2,
      518,
      518
    ]
  },
  "identity_off_parity": "PASS",
  "protocol_id": "H2_CAB_LORA_R1_BOUNDED",
  "source_batch_file_names": [
    "pcb1/Data/Images/Normal/0417.JPG"
  ],
  "strict_tolerance": 1e-06
}
