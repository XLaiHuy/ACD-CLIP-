# Pre-edit Phase2B / H6 / dataset audit

Audit base: `1bfe19ca158ff70a79f0861483e6b18bb1b7341f`
Worktree: `/home/ai4/caohuy/ACD-CLIP-sabra-canonical-v1`

## Boundary findings

- The existing adapter is `model.adapter.ACDCLIP`. Its legitimate Phase2B
  state is the image/text adapter modules, DFG configuration, soft/hybrid
  prompt path, and the detection/segmentation projections. The optimizer
  groups in the historical trainer are `image_adapter` and `text_adapter`.
- Historical H6 is optional construction in `ACDCLIP` and is threaded through
  the root `train.py`, root `test.py`, and checkpoint helpers. The current
  public entrypoints contain H6 CLI flags, H6 checkpoint auto-detection, H6
  batch construction, and H6 residual logits. This is contamination of the
  canonical entrypoint graph, not a reason to delete `model/h6/`.
- The canonical runtime must construct the existing adapter with the legacy
  compatibility switch disabled, then assert that no H6 module/output is
  active or consumed. Legacy construction details belong in one bridge.

## Exact Phase2B forward contract from source

1. `ACDCLIP.forward(..., return_phase4_features=True)` returns `seg_tokens`
   and `det_tokens`. With the audited three-group setup, each segmentation
   token tensor is `[B, 1369, 768]`; stacked stage features are
   `[3, B, 1369, 768]`. Detection features are `[3, B, 768]`.
2. `get_phase2b_global_text_features` returns text features
   `[3, B, 768, 2]`, using the selected prompt path.
3. `vision_text_fusion_gate_seg` applies DFG weights, multiplies each stage's
   image tokens by its two-class text features, and returns native stage
   logits `[3, B, 1369, 2]` plus native margin
   `abnormal_logit - normal_logit` with shape `[3, B, 1369]`.
4. Native deployment is Gaussian blur (industrial kernel 7, sigma 1),
   bilinear resize to `518 x 518` with `align_corners=True`, mean over the
   three stages, then two-class softmax. `deploy_with_delta` adds a delta to
   native logits before this same operator.
5. Classification uses detection/text products `[3, B, 2]`, averages stages,
   then applies two-class softmax. SABRA must not modify this branch.

## Checkpoint and field audit

The current adapter checkpoint loader consumes `image_adapter`,
`text_adapter`, and optional `soft_prompt` state. Runtime configuration
needed by the adapter/CLIP constructor is limited to model name, image size,
group count, adapter/LoRA dimensions and weights, DFG fields, prompt fields,
and precision/checkpointing policy. H6 metadata, losses, routers, factors,
experts, rho, and diagnostics are historical or diagnostic-only and are not
part of the canonical config.

The canonical config will retain source-owned fields for model construction,
prompt path, DFG, LoRA/adaptation, and training policy. It will explicitly
classify removed historical fields instead of copying the Phase4 config.

## Dataset roles

- `dataset/info.py` maps VisA and MVTec to the Industrial domain and Medical
  datasets to the Medical domain. The official MVTec class inventory is the
  15-class list in `CLASS_NAMES["MVTec"]`; `dataset/hub/MVTec.jsonl` carries
  image, label, mask, and class identity.
- The generic dataset package can load image+GT records for evaluation. The
  existing `tools/sabra/data.py` supplies a separate deterministic
  `VisaEvidenceDataset` that exposes only image/class/path and never opens or
  stores labels or masks. This is the authoritative GT firewall input for
  relational record construction; `VisaEvaluationDataset` is separate and
  only joins GT after records exist.
- Medical roots are configured by `MEDICAL_ROOT`/`ACDCLIP_DATA_ROOT` and the
  Medical evaluation path map. They are final-test-only and must be rejected
  by training/calibration entrypoints.

## Implementation consequence

The canonical public graph will be `train.py` / `test.py` / selector /
calibration -> shared Phase2B runtime -> optional legacy bridge -> existing
ACDCLIP base, followed by shared evaluator and explicit SABRA relational,
Trust, Need, Authority, and correction modules. No real training, MVTec
forward, calibration, lambda sweep, or Medical inference is permitted during
this setup task.
