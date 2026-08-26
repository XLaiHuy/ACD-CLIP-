# P29 — SABRA Sign-Guarded Normalized Region Distillation V1

P29 is a preregistered VisA-development falsification follow-up to P28R1. It
changes only the adapter training objective; it neither selects a deployable
architecture nor supplies independent external validation.

## Frozen method

The P27 `RegionResidualAdapter`, R0 teacher, 37×37 patch grid, 9×9
adaptive-average region target, three frozen visual stages, symmetric margin
integration, P26 deployment, LOCO class order, AdamW schedule, FP32 policy,
and exact Tier-A cache contract are retained unchanged. The fixed correction
scale is `C = 4.960109710693359`.

For normalized teacher and student regions `t = teacher/C` and `s = student/C`,
the objective is exactly:

```
SmoothL1(s, t) + mean(abs(t) * relu(-sign(t) * s))
+ mean_over_pure_normal(relu(s)^2)
```

`pure_normal` is the complement of source-mask occupancy after deterministic
adaptive max pooling from image support to 37×37 then 9×9. No segmentation
localization loss, target clipping, output clipping, threshold, scaling fit,
capacity increase, ranking loss, feature loss, memory/prototype bank,
attention, or sweep is permitted.

## Scientific firewall

Each LOCO fold trains on its 11 source classes only. Before the held prediction
is frozen, held GT, mask, teacher, and normal-region targets are unread. All
12 predictions freeze before any scoring. MVTec and Medical are prohibited.

## Decision contract

`P29_SUPPORTED` requires macro pAP above native, macro pAUROC at least native,
positive median pAP delta, pAP improvement in at least 7 of 12 classes, and a
passing audit. Otherwise P29 is terminalized as mixed/not-supported according
to the preregistered evidence; no P29R2 or next phase is started automatically.
