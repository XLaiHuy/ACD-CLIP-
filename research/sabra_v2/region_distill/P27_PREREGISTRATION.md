# P27 Preregistration — SABRA-V2 Region Correction Distillation V1

## Scope

This is one fixed source-only architectural experiment: distill R0-style
signed correction behavior into a compact 9x9 residual representation that
improves native Phase2B logits directly. It does not reinterpret P26's result
that runtime patch-benefit control was not established.

## Parent identities

- P26 handoff parent: `ae2388f7cd96a60a92fae75bd1c0f9e11284fd28`
- P26 scientific architecture: `a28b7839f65376eb9c422f36a5e7f7a18f992012`
- P26 tag: `sabra-final-p26-v1`

## Frozen design

- Feature inputs: Phase2B L8/L16/L24 segmentation features `[3,B,1369,768]`
  and native logits `[3,B,1369,2]`.
- Region grid: deterministic adaptive average pooling from 37x37 to 9x9.
- Student: shared 768-to-64 projection; concatenated stage context; 1x1,
  depthwise 3x3, 1x1 convolutional residual head producing three maps.
- Residual: bilinear 9x9-to-37x37; symmetric two-class margin correction.
- Teacher: exact R0 signed-utility action semantics; source GT only; fixed
  historical alpha 0.25 and R0 margin scale, used only to create targets.
- Loss: equal-weight SmoothL1 region distillation plus canonical focal/dice
  localization loss.
- Trainable parameters: P27 region adapter only.

## LOCO protocol

VisA classes are `candle`, `capsules`, `cashew`, `chewinggum`, `fryum`,
`macaroni1`, `macaroni2`, `pcb1`, `pcb2`, `pcb3`, `pcb4`, and `pipe_fryum`.
Each scientific fold holds out one class. The held class must not enter fit
inventory, teacher construction, masks, GT, statistics, selection, or tuning.

## Evaluation and interpretation

The future scientific run reports macro and per-class pixel AP/AUROC from
held predictions. A useful region teacher should retain meaningful R0-style
headroom; a useful student should improve broad LOCO pAP without material
AUROC damage. These are practical scientific interpretations, not brittle
decimal gates.

`REGION_TEACHER_HEADROOM=NOT_CHECKED` for this execution base. Historical R0
caches use an incompatible checkpoint and must not be used to infer a P26
teacher-headroom result.

## Stop conditions and firewall

This machine performs no full scientific 12-fold run. The frozen 9x9 grid,
student family, teacher semantics, losses, and LOCO protocol cannot be changed
after engineering smoke without explicit review. MVTec is never opened or
read; Medical is never read; target inference never accepts GT or masks.
