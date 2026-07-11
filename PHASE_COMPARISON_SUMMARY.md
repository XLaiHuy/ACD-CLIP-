# Phase Comparison Summary

This is an exploratory local comparison. Pixel-level means are macro means over
six medical datasets: Brain, Liver, Retina, Colon_clinicDB, Colon_colonDB, and
Colon_Kvasir. Image-level means are macro means over Brain, Liver, and Retina
only; Colon image metrics are not included because those test splits do not
contain both image labels.

## Overall Table

| Phase | Version / run | Epoch | Image score | Pixel mean 6 AUC/AP | Image mean 3 AUC/AP | Note |
|---|---|---:|---|---:|---:|---|
| Baseline | mainbase best pixel AP | 9 | current | 89.77 / 38.24 | 73.33 / 73.89 | Local reproduced baseline pixel-best |
| Baseline | mainbase best pixel AUC | 10 | current | 90.25 / 38.21 | 73.80 / 74.36 | Baseline stronger image reference |
| Phase 1A | V1 tau4 best pixel | 15 | current | 87.45 / 31.47 | - | Below baseline |
| Phase 1A | V1 tau4 best image | 17 | current | - | 73.38 / 74.60 | Image improves, pixel not selected |
| Phase 1A | V2 tau8 best pixel | 13 | current | 89.63 / 34.70 | 72.23 / 73.11 | Slightly better than V1, below baseline AP |
| Phase 1A | V2 tau8 best image | 10 | current | 89.72 / 34.32 | 72.65 / 73.58 | Below baseline |
| Phase 1B | SS2D g02 best pixel | 11 | current | 90.23 / 34.77 | 73.23 / 72.58 | AUC improves, AP still weak |
| Phase 1B | SS2D g02 best image | 12 | current | 88.17 / 30.57 | 74.08 / 74.24 | Image improves, pixel drops |
| Phase 1B | V3c old before fp32 | 9 | current | 90.91 / 35.63 | - | Better pixel AUC, AP below final |
| Phase 1B | V3c fp32 final anchor | 9 | current | 90.76 / 39.82 | 73.80 / 75.06 | Best Phase 1 balanced anchor |
| Phase 2B | Hybrid alpha0.2, no K-reg | 9 | current | 89.87 / 39.54 | 72.70 / 74.13 | Near Phase 1 pixel AP, weaker image |
| Phase 2B | Hybrid alpha0.1 + K-reg 5e-3 | 7 | current | 90.52 / 37.81 | 72.70 / 73.03 | Conservative; does not beat Phase 1 |
| Phase 2B | Hybrid alpha0.2 + K-reg 2e-3 | 10 | current 0.5 cls + 0.5 max | 90.98 / 40.35 | 71.65 / 70.71 | Best pixel AP, image score harmed |
| Phase 2B | Hybrid alpha0.2 + K-reg 2e-3 | 10 | cls_only | 90.98 / 40.35 | 73.77 / 74.24 | Best overall under pixel-first constraint |
| Phase 2B | Hybrid alpha0.2 + K-reg 2e-3 | 11 | cls_only | 90.47 / 39.06 | 73.87 / 74.46 | Better image than e10, lower pixel |
| Phase 2B | Hybrid alpha0.2 + K-reg 2e-3 | 14 | cls_only | 89.29 / 36.13 | 76.19 / 76.86 | Best image, pixel gain lost |

## Phase 2B Fixed `cls_only` Sweep

| Epoch | Pixel mean 6 AUC/AP | Image mean 3 AUC/AP | Selection note |
|---:|---:|---:|---|
| 7 | 89.61 / 37.00 | 72.96 / 73.10 | Below image constraint |
| 8 | 90.00 / 38.29 | 72.35 / 72.99 | Below image constraint |
| 9 | 89.88 / 37.75 | 73.37 / 73.71 | Just below image constraint |
| 10 | 90.98 / 40.35 | 73.77 / 74.24 | Best selected checkpoint |
| 11 | 90.47 / 39.06 | 73.87 / 74.46 | Acceptable, less pixel AP |
| 12 | 90.12 / 36.59 | 74.58 / 75.31 | Image good, pixel drops |
| 13 | 90.00 / 36.27 | 74.78 / 75.44 | Image good, pixel drops |
| 14 | 89.29 / 36.13 | 76.19 / 76.86 | Best image, pixel drops |
| 15 | 89.09 / 36.36 | 75.47 / 76.18 | Image good, pixel drops |

## Best Overall Choice

Using the selection rule:

```text
maximize pixel AP over 6 datasets
subject to image AP over 3 datasets >= 73.80
tie-break by image AP
```

the best overall checkpoint is:

```text
Phase2B Hybrid alpha0.2 + K-reg 2e-3
epoch 10
image score = cls_only
pixel mean 6 AUC/AP = 90.98 / 40.35
image mean 3 AUC/AP = 73.77 / 74.24
```

Compared with the local baseline pixel-best checkpoint:

```text
Pixel AUC/AP: +1.21 / +2.11
Image AUC/AP: +0.44 / +0.35
```

Compared with the Phase 1 final anchor:

```text
Pixel AUC/AP: +0.22 / +0.53
Image AUC/AP: -0.03 / -0.82
```

Interpretation: Phase2B e10 with `cls_only` image scoring is the best
pixel-first balanced result. Phase1 V3c fp32 final remains the cleaner
image-stability anchor, but it gives up the Phase2B pixel AP gain.

## Per-Dataset Metrics For Selected Overall Best

Phase2B Hybrid alpha0.2 + K-reg 2e-3, epoch 10, `cls_only` image score:

| Dataset | Pixel AUC/AP | Image AUC/AP |
|---|---:|---:|
| Brain | 95.24 / 38.28 | 82.10 / 94.78 |
| Liver | 97.07 / 6.81 | 63.40 / 57.53 |
| Retina | 92.20 / 40.39 | 75.81 / 70.40 |
| Colon_clinicDB | 89.06 / 57.46 | - |
| Colon_colonDB | 83.88 / 35.70 | - |
| Colon_Kvasir | 88.40 / 63.45 | - |
| **Mean** | **90.98 / 40.35** | **73.77 / 74.24** |

## Main Finding

The Phase2B image-level drop was mainly caused by image-score aggregation, not
by prompt split. On e10, all prompt configurations gave nearly identical
`cls_only` image AP, while adding pixel score reduced image AP:

| Prompt config | Score rule | Pixel mean 6 AUC/AP | Image mean 3 AUC/AP |
|---|---|---:|---:|
| current_shared | cls_only | 90.98 / 40.35 | 73.77 / 74.24 |
| split_hard_cls | cls_only | 90.98 / 40.35 | 73.77 / 74.24 |
| split_lowalpha_cls | cls_only | 90.98 / 40.35 | 73.77 / 74.24 |
| current_shared | 0.5 cls + 0.5 max | 90.98 / 40.35 | 71.63 / 70.72 |
