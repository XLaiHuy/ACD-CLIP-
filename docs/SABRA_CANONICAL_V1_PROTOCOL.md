# SABRA canonical v1 protocol

This document freezes the setup boundary. It does not authorize a real
training, MVTec evaluation, calibration, lambda sweep, or Medical run.

## Phase2B field ownership

`model_name`, `img_size`, `precision`, and `grad_checkpointing` are consumed
by the CLIP constructor/runtime. `n_groups`, adapter weights, LoRA ranks and
alphas, and convolution kernel sizes are consumed by the ACD-CLIP adapter.
The `dfg_*` fields are consumed by the Phase2B DFG implementation.
Prompt fields are consumed by the hard/soft hybrid prompt path. The training
fields are consumed only by the Phase2B VisA trainer and are persisted in the
run manifest.

The removed Phase4 fields are historical routing/factor/VAE/expert losses,
legacy residual coefficients, diagnostic gates, rho controls, and checkpoint
telemetry. They are not copied into the canonical config. Historical
`model/h6/` sources remain in the repository but are unreachable from the
canonical public graph.

## Input -> output flow

STAGE 1

INPUT
  VisA
  Phase2B canonical config
  CLIP asset

PROCESS
  train Phase2B

OUTPUT
  adapter_10.pth
  adapter_12.pth
  adapter_14.pth
  adapter_16.pth
  adapter_18.pth
  adapter_20.pth
  train_manifest.json

PHASE2B DEVELOPMENT SELECTION

INPUT
  candidate checkpoints
  MVTec AD

PROCESS
  exact evaluation
  S=.35pAUROC+.35pAP+.15iAUROC+.15iAP

OUTPUT
  phase2b_selection.json

SABRA SOURCE CALIBRATION

INPUT
  selected E*
  VisA

PROCESS
  frozen Phase2B
  relational E
  Trust logistic
  Need intervention oracle + logistic
  s_m

OUTPUT
  sabra_source_calibration.json

SABRA LAMBDA DEVELOPMENT

INPUT
  sabra_source_calibration.json
  MVTec AD

PROCESS
  frozen Phase2B/Trust/Need
  lambda sweep only

OUTPUT
  SABRA_FREEZE.json

FINAL ZERO-SHOT TEST

INPUT
  SAME E*
  SAME SABRA_FREEZE
  Medical

PROCESS
  native Phase2B
  vs
  SABRA corrected Phase2B

OUTPUT
  native metrics
  SABRA metrics
  deltas

## Runtime boundaries

The one shared runtime returns segmentation features, detection features,
text features, native logits, native margin, native deployment, and the
classification probability. Native logits have shape `[3, B, 1369, 2]`.
Deployment is Gaussian blur, aligned bilinear resize, stage mean, and
two-class softmax. SABRA adds only a normal-zero/abnormal-positive delta to
these native logits; the classification branch is unchanged.

VisA is the only Stage 1 and source-calibration dataset. MVTec is development
data for checkpoint/lambda selection. Medical is final zero-shot test only.
