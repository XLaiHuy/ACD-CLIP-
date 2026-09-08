# H2 NFUR-R2 final report

{
  "branch": "research/h2-nfur-r2-e20-medical-selected",
  "compute_overhead": {
    "head": "Conv2d(772,32,3)+GELU+Conv2d(32,1,1)",
    "parameters": 222401
  },
  "final_interpretation": "MIXED_TRADEOFF",
  "mechanism": "small native score-space residual addresses contextual coupling while preserving exact OFF identity; no finer-than-37x37 feature was available",
  "medical": {
    "delta_vs_h2_anchor_a_e15": {
      "image_ap": -0.4017353057861328,
      "image_auroc": -0.6672700246175225,
      "pixel_ap": -1.326754791512748,
      "pixel_auroc": -3.0900368218221246
    },
    "medical_untouched_zero_shot_claim_allowed": false,
    "phase2b_comparison_caveat": "descriptive only: older rounded pixel_stride=4 path versus raw exact pixel_stride=1 H2 path",
    "phase2b_context": {
      "image_ap": 74.24,
      "image_auroc": 73.77,
      "pixel_ap": 40.35,
      "pixel_auroc": 90.98
    },
    "selected": {
      "dataset": "ALL_MEDICAL_MACRO",
      "dataset_count": 6,
      "domain": "Medical",
      "epoch": 6,
      "image_ap": 75.94162821769714,
      "image_auroc": 74.61545666058858,
      "image_dataset_count": 3,
      "inference": "equal",
      "pixel_ap": 38.141614904804015,
      "pixel_auroc": 88.1617492355726,
      "scope": "macro",
      "valid": true
    },
    "selected_epoch": 6,
    "target_validation_used": true,
    "trajectory_artifact": "results/H2_NFUR_R2_E1_E20_MEDICAL.json"
  },
  "medical_artifact": "/workspace/ACD-CLIP-/results/H2_NFUR_R2_E1_E20_MEDICAL.json",
  "mvtec": {
    "delta_vs_h2_anchor_a_e15": {
      "image_ap": 0.2722015808308811,
      "image_auroc": -0.5738522522176055,
      "pixel_ap": -1.439117740579512,
      "pixel_auroc": 0.5432893457295762
    },
    "holdout_only_after_medical_selection": true,
    "selected": {
      "dataset": "ALL_MVTEC_MACRO",
      "dataset_count": 15,
      "domain": "MVTec",
      "epoch": 6,
      "image_ap": 95.05156358083089,
      "image_auroc": 89.2430607477824,
      "image_dataset_count": 15,
      "inference": "equal",
      "pixel_ap": 43.72023125942049,
      "pixel_auroc": 90.58457834572958,
      "scope": "macro",
      "valid": true
    }
  },
  "mvtec_artifact": "/workspace/ACD-CLIP-/results/H2_NFUR_R2_MVTEC_FINAL.json",
  "nfur": {
    "added_parameters": 222401,
    "design": "NFUR-R2",
    "formulation": "equal-stage native margin disagreement plus low fused-margin uncertainty gates a zero-initialized bounded local residual",
    "implemented": true
  },
  "parent_head": "284b12b6dc7802bcbafc02d7809549c40758f94b",
  "protocol_id": "H2_NFUR_R2_E20_MEDICAL_SELECTED",
  "published_surpass_claim_allowed": false,
  "published_surpass_claim_why": "Medical epoch selection used target validation and Phase2B uses a non-identical older evaluator; MVTec is a post-selection holdout, so no clean published-surpass claim is authorized."
}
