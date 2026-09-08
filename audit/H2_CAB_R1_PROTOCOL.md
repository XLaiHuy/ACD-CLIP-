# H2 CAB-LoRA R1 Protocol

{
  "arms": [
    "CONTROL",
    "CAB-LoRA R1"
  ],
  "branch": "research/h2-cab-lora-r1-bounded",
  "cab_disabled_default": true,
  "cab_target_gradient_ratio": 0.025,
  "cab_target_gradient_ratio_rationale": "2.5% is a single conservative source-only calibration target: CAB acts at representation level upstream of logits, and a small ratio reduces the risk of suppressing useful anomaly-context relationships; it is not the prior S2-LOCR 5% value.",
  "counterfactual": "cyclic peer context from the same source batch, with union of selected token footprints plus 14px halo protected and a 14px distance-based smooth transition",
  "formula": "mean_i ReLU((1-cos(z_post(x_i),z_post(x_tilde_i)))-(1-cos(z_pre(x_i),z_pre(x_tilde_i))))",
  "full_train_justified_only_if_bounded_pass": true,
  "intervention_scope": [
    "Stage-2 Conv-LoRA only"
  ],
  "matched_contract": [
    "starting checkpoint",
    "seed",
    "attempt manifest",
    "batch order",
    "augmentations",
    "optimizer",
    "LR",
    "scheduler",
    "AMP",
    "gradient clipping",
    "Safe Anchor",
    "DFG",
    "SS2D",
    "prompt settings",
    "attempt count"
  ],
  "max_attempts": 500,
  "mechanism": "CAB-LoRA R1",
  "native_grid": [
    37,
    37
  ],
  "no_medical_tuning": true,
  "no_mvtec_tuning": true,
  "no_sweep": true,
  "parent_head": "284b12b6dc7802bcbafc02d7809549c40758f94b",
  "patch_footprint": [
    14,
    14
  ],
  "post_adapter_tensor": "norm-matched Stage-2 Conv-LoRA output merged with detached m_i_w[1] coefficient",
  "pre_adapter_tensor": "Stage-2 Conv-LoRA input t=x[1:] after transformer block 16, before image_adapter.lora_adapters[1]",
  "protected_geometry": "native token footprint rows [14r,14r+13], cols [14c,14c+13]",
  "protocol_id": "H2_CAB_LORA_R1_BOUNDED",
  "selected_tokens": "up to 4 deterministic row-major valid near-background tokens per image; zero-footprint and intersecting existing 7x7 near-background region",
  "source_dataset": "VisA train",
  "target_inference_before_decision": false,
  "unchanged_modules": [
    "Stage-1 Conv-LoRA",
    "Stage-3 Conv-LoRA",
    "DFG",
    "DFG Q/K",
    "SS2D",
    "segmentation projection",
    "prompt branch",
    "stage fusion",
    "inference smoothing",
    "interpolation",
    "NFUR",
    "classification head",
    "evaluation code"
  ]
}
