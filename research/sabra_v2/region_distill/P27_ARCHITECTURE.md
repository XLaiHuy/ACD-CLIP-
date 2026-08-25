# P27 SABRA-V2 Region Correction Distillation V1

## Frozen parent

P27 is derived from the P26 portable handoff `ae2388f7cd96a60a92fae75bd1c0f9e11284fd28` and preserves the P26 scientific architecture SHA `a28b7839f65376eb9c422f36a5e7f7a18f992012`. P26 files and behavior remain immutable.

## Runtime graph

The frozen Phase2B model exposes normalized segmentation features
`[3, B, 1369, 768]` for the canonical L8/L16/L24 stages and native logits
`[3, B, 1369, 2]`. P27 reshapes the features to 37x37, applies a shared
768-to-64 projection to each stage, and adaptively averages each stage to a
fixed 9x9 grid. The three pooled stage tensors are concatenated, processed by
a 1x1 convolution, GELU, depthwise 3x3 convolution, GELU, and a 1x1
three-channel output head. This yields `[3, B, 9, 9]` region-margin residuals.

Each residual is bilinearly upsampled to 37x37 and integrated before the
unchanged Phase2B deployment operator. For each stage and patch, scalar delta
is applied symmetrically:

```text
normal_logit  <- normal_logit  - delta / 2
anomaly_logit <- anomaly_logit + delta / 2
```

Thus delta changes anomaly-minus-normal margin by exactly delta without adding
a common logit offset. There is no inference-time oracle, alpha predictor, or
BOOST/SUPPRESS/KEEP controller.

## Source-only teacher

For a source training batch only, P27 reuses R0's signed utility convention:
the negative gradient of canonical focal/dice localization loss with respect
to the shared abnormal-logit correction. Utility above/below R0 epsilon maps
to BOOST/SUPPRESS, otherwise KEEP. The historical teacher provenance is the
signed correction `action * 0.25 * R0_MARGIN_SCALE` applied to the abnormal
channel, shared across stages, before Phase2B deployment.

P27 explicitly converts that historical one-channel teacher correction to the
symmetric student-margin target by comparing corrected and native margins.
The patch target `[B,37,37]` is adaptively averaged to `[B,9,9]`. GT masks are
used only in this teacher construction and source localization loss. Student
inference consumes only frozen visual features and native logits.

## Training graph

All Phase2B components are frozen: CLIP, image adapter, text adapter, soft
prompt, canonical preprocessing, and deployment/postprocessing. The only
trainable module is `RegionResidualAdapter`.

The fixed P27 V1 objective is:

```text
SmoothL1(predicted_region_delta, source_teacher_region_delta)
+ canonical_focal_dice(corrected_source_probability, source_mask)
```

Both terms have weight 1.0. No coefficient or region-size sweep is permitted.

## Data protocol

P27 scientific execution uses VisA only with 12-class LOCO. For a held class,
all teacher generation, fitting, normalization, and checkpoint selection use
only the other 11 classes. Held GT is unavailable to training interfaces. The
held class is evaluated only after predictions are frozen. MVTec remains
sealed and Medical is forbidden.
