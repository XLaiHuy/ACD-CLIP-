# H2 Boundary Spillover R1 — Score-Map Provenance

## Frozen endpoint and inference identity

Primary endpoint: Safe-Anchor E10, loaded read-only from `/workspace/h2_safe_anchor_e20_medical_selected/adapter_10.pth`
(SHA-256 `64b72dc3d1155285c826781bee4c5970bd45218e95b21675fd19d9a6b2ab54a7`). The frozen cohort is
`audit/H2_FUSION_ENDPOINT_EVAL_SUBSET.csv`, exactly 96 images, 8 per VisA
category, with the cohort artifact recording source image/mask hashes. The
prior freeze and THBR endpoints are used only as descriptive inference-only
comparison arms in the final triangulation.

Model construction is the committed H2 Safe-Anchor E10 architecture and
configuration: ViT-L/14-336, image size 518, three stages, DFG attention with
SS2D weight residual, hybrid text prompts, and equal stage fusion. No
architecture, feature detachment, fusion weight, or interpolation change is
made by this audit. `eval()`, `requires_grad_(False)`, `torch.no_grad()`, and
the historical FP16 autocast inference policy are used; there is no backward or
optimizer operation.

## Map inventory

* Stage 1/2/3 native anomaly maps: each stage's native 37x37 class logits from
  `_vision_text_attention_fusion`, converted with per-stage softmax. Native
  logits are retained conceptually for the equal pre-softmax native combined
  map.
* Resized stage maps: each native stage logit map receives the existing VisA
  test-mode 7x7 Gaussian blur (`sigma=1`) and is then bilinearly resized to
  518x518 with `align_corners=True`; per-stage softmax yields the stage anomaly
  probability.
* Pre-fusion combined map: equal mean of the three native stage logits followed
  by softmax for native-resolution descriptive measurements.
* Final resized/evaluator map: equal mean of the three resized, blurred stage
  logits followed by softmax. This is the current production VisA endpoint path
  in `vision_text_fusion_gate_seg(test_mode=True, domain="Industrial")`. There
  is no separate hidden final resize after this map.
* Fusion attribution additionally reports the equal pre-softmax fused anomaly
  logit margin (`logit_abnormal - logit_normal`) so that the pre-softmax map
  remains in its native score space rather than being silently redefined as a
  probability.

All interpolation statements are observational comparisons of the exact
current path; no alternative interpolation is evaluated or selected.

## Prior evidence recorded for triangulation

* `research/h2-late-convlora-freeze-r1` at `4eb3e134e2126afa980fa15f1e7b5a9da9d9141e`:
  `MECHANISM_TEST=SUPPORTED`, `BOUNDED_SCREEN=FAIL`, and
  `LATE_CONVLORA_CAUSAL_HYPOTHESIS=NOT_SUPPORTED`. Its candidate-minus-control
  near-background p95/p99 deltas were `+0.03233134001493454` and
  `+0.05800473690032959`, while AP delta was `-0.0058612052382776`.
  Its retained final checkpoint is `/workspace/h2_late_convlora_freeze_r1/A_LATE_FREEZE_R1_STAGE23_CONVLORA/final.pth`, SHA-256
  `4a5f3fd0e54b53f8cc78060a4092e3d1240792f2809d7ebfe57e9340789fc9f8`.
* `research/h2-thbr-r1` at `5d84c3d857f43ae3f7d39210fff2faf983429513`:
  `DECISION=FAIL`, `HARD_BACKGROUND_TAIL_PREMISE=SUPPORTED`,
  `BOUNDARY_ARTIFACT_RISK=LOW`, and `FOCAL_REDUNDANCY=LOW`. Its candidate-minus-
  control near-background p95/p99 deltas were `+0.008221146836876858` and
  `+0.06695230543613492`, with AP delta `-0.010847331119427928`; its audit
  explicitly says far-background improved while near-background worsened.
  The retained THBR control and candidate endpoint checkpoints are identified
  by SHA-256 `3c337dc581aabd491f05e9788e07711963aebd6bee7379f92156e206fd8cf0b8`
  and `96f2a93fc2018b16d6376a9c6933ff9512f99ab1db94ce0e971a1942a39390ba`.
* `audit/H2_THBR_R1_AUDIT_DECISION.md` on that evidence branch records a
  source-only audit, the existing 7x7 semantics, and that the numerically
  invalid GradBudget endpoint is descriptive only. No GradBudget endpoint is
  used as primary evidence here.

## Explicit map contract

| map | native tensor / score tensor | spatial resolution | score space | logits or probabilities | interpolation afterward | fusion afterward |
|---|---|---|---|---|---|---|
| stage1 native | `[B,2,37,37]` / `[B,37,37]` | 37x37 | class anomaly score | native logits; stage anomaly probability for regional statistics | none for native audit; current production branch applies Gaussian7 then bilinear518 | none as an individual stage |
| stage2 native | `[B,2,37,37]` / `[B,37,37]` | 37x37 | class anomaly score | native logits; stage anomaly probability for regional statistics | none for native audit; current production branch applies Gaussian7 then bilinear518 | none as an individual stage |
| stage3 native | `[B,2,37,37]` / `[B,37,37]` | 37x37 | class anomaly score | native logits; stage anomaly probability for regional statistics | none for native audit; current production branch applies Gaussian7 then bilinear518 | none as an individual stage |
| pre-fusion combined | `[B,2,37,37]` / `[B,37,37]` | 37x37 | equal fused class logit / probability | equal mean of native stage logits, then softmax | none in native diagnostic map | equal pre-softmax mean of stage logits |
| resized stage1/2/3 | `[B,2,518,518]` / `[B,518,518]` | 518x518 | class anomaly score | blurred/resized logits; stage anomaly probability | existing Gaussian7 sigma1 then bilinear, `align_corners=True` | none as an individual stage |
| final fused | `[B,2,518,518]` / `[B,518,518]` | 518x518 | equal fused class logit margin / anomaly probability | equal mean of resized stage logits; softmax probability | no further resize | equal pre-softmax mean |
| final resized/evaluator | `[B,518,518]` | 518x518 | anomaly probability | alias of final fused production output | no further resize | already equal-fused; this is the current evaluator output |

The `final fused` and `final resized/evaluator` rows intentionally refer to
the same current production output; the audit does not create a second final
prediction.
