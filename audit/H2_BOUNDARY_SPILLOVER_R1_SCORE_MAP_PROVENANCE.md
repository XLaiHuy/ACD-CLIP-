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
