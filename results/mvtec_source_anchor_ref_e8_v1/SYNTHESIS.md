# E8-reference experiment synthesis

## Confirmatory VisA-source experiment

The frozen VisA-source E8-reference Anchor result at E15 was:

| Evaluation | Pixel AUROC | Pixel AP | Image AUROC | Image AP |
|---|---:|---:|---:|---:|
| Medical E8-reference Anchor | 89.242041 | 36.243819 | 76.198479 | 75.997533 |
| Medical H E15 | 90.815332 | 35.874253 | 76.148768 | 76.250001 |
| Medical Anchor-to-E1 E15 | 91.251786 | 39.468370 | 75.282727 | 76.343364 |
| MVTec E8-reference Anchor | 84.588163 | 41.524673 | 89.150273 | 95.054912 |
| MVTec H E15 | 86.868623 | 41.612306 | 89.526111 | 95.207364 |
| MVTec Anchor-to-E1 E15 | 90.041289 | 45.159349 | 89.816913 | 94.779362 |

Relative to the existing Anchor-to-E1 E15 endpoint, the E8-reference endpoint changes Medical by `-2.009745 / -3.224551 / +0.915752 / -0.345830` percentage points and MVTec by `-5.453126 / -3.634676 / -0.666640 / +0.275550` percentage points, in the column order Pixel AUROC / Pixel AP / Image AUROC / Image AP.

## Decision

The E8 reference does not clearly win. The result is mixed on image-level metrics and is negative on both pixel measures against Anchor-to-E1; versus H, Medical Pixel AP is slightly higher while the other pixel comparison is lower. The practical recommendation is to retain the existing E1-reference Anchor as the default, and treat the mature-clean-E8 reference as an informative negative/mixed ablation rather than a replacement strategy.

This recommendation is qualified: Anchor-to-E1 had a longer Anchor continuation than Anchor-to-E8, so the comparison does not perfectly isolate reference age from Anchor exposure duration.

## Source-dataset reversal

The complementary MVTec-source run used the exact audited 1,725-row MVTec pool and evaluated on VisA. Its pre-specified E15 result was `92.735696 / 18.386151 / 81.464077 / 85.069036`; E10 was diagnostically stronger on all four metrics, but was not promoted. Full trajectory and per-class details are in [`REPORT.md`](REPORT.md). These reversal numbers are not substituted into the matched VisA-source Medical/MVTec comparison above because source and target domains are reversed.
