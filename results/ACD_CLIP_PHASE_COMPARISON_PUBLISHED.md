# ACD-CLIP Phase 2 / H2 Results Publication

Published research snapshot: 2026-09-04.

This report consolidates the frozen H2 four-arm factorial, its Medical and
MVTec evaluations, and a descriptive comparison with the locally stored Phase
2 results. H2 E15 is primary and H2 E20 is secondary. The raw H2 tables,
freeze manifest, and protocol audit remain the authoritative evidence.

## Result in one sentence

The preregistered minimal-winner rule selects **A (Anchor)**: at E15, A
Pareto-beats H on the six-dataset Medical pixel metrics by `+0.436` AUROC and
`+3.594` AP, while C fails and AC does not beat H or A. E20 preserves the AP
gain but not the AUROC gain.

## Protocol and validity

- H2: fresh shared E1, four arms H/A/C/AC, E15 primary and E20 secondary.
- Medical: six datasets, `benchmark_exact`-equivalent evaluation,
  `current_shared`, `cls_only`, raw exact metrics, `pixel_stride=1`, no
  checkpoint selection and no tuning.
- MVTec: E15 matched industrial transfer confirmation under the same exact
  evaluation constraints; it was not used to retune the Medical decision.
- The later H/A seed-1 and seed-2 replication attempts are **invalid and have
  no Medical/MVTec metrics** because each had a global-step mismatch and
  non-finite-gradient skips. They are not pooled with the eligible H2 seed-0
  results.

## Medical macro results

Values are `pixel AUROC / pixel AP` and `image AUROC / image AP`. Image means
use Brain, Liver, and Retina only because the Colon test splits do not contain
both image labels.

| Horizon | Arm | Pixel mean 6 | Image mean 3 |
|---:|---|---:|---:|
| E15 | H | 90.815 / 35.874 | 76.149 / 76.250 |
| E15 | A | **91.252 / 39.468** | 75.283 / 76.343 |
| E15 | C | 90.220 / 34.493 | 75.515 / 75.465 |
| E15 | AC | 90.801 / 37.054 | **76.614 / 76.699** |
| E20 | H | 90.999 / 35.640 | 76.376 / 76.437 |
| E20 | A | 90.991 / **38.548** | 75.788 / 76.501 |
| E20 | C | 90.304 / 35.210 | **76.470 / 76.756** |
| E20 | AC | 90.660 / 37.965 | 76.346 / 76.364 |

### Per-dataset Medical results: E15 primary

Cells are `pixel AUROC / pixel AP`.

| Dataset | H | A | C | AC |
|---|---:|---:|---:|---:|
| Brain | 94.683 / 32.076 | 95.459 / 36.196 | 94.481 / 30.868 | 94.521 / 30.314 |
| Liver | 96.712 / 6.109 | 96.380 / 5.467 | 96.596 / 6.638 | 96.128 / 5.308 |
| Retina | 94.686 / 47.596 | 95.360 / 55.069 | 92.869 / 41.417 | 93.609 / 46.020 |
| Colon_clinicDB | 86.982 / 43.964 | 88.657 / 50.466 | 87.764 / 47.586 | 89.467 / 50.694 |
| Colon_colonDB | 86.160 / 32.336 | 85.054 / 35.475 | 84.350 / 28.885 | 84.168 / 33.716 |
| Colon_Kvasir | 85.669 / 53.164 | 86.601 / 54.136 | 85.262 / 51.564 | 86.915 / 56.273 |

Per-dataset deltas against H at E15 (`AUROC / AP`):

| Dataset | A - H | C - H | AC - H |
|---|---:|---:|---:|
| Brain | +0.777 / +4.120 | -0.202 / -1.208 | -0.162 / -1.762 |
| Liver | -0.332 / -0.642 | -0.116 / +0.528 | -0.584 / -0.802 |
| Retina | +0.673 / +7.473 | -1.818 / -6.179 | -1.077 / -1.576 |
| Colon_clinicDB | +1.675 / +6.502 | +0.782 / +3.622 | +2.485 / +6.730 |
| Colon_colonDB | -1.106 / +3.140 | -1.810 / -3.451 | -1.992 / +1.380 |
| Colon_Kvasir | +0.932 / +0.972 | -0.408 / -1.600 | +1.245 / +3.109 |

A improves both pixel metrics on Brain, Retina, Colon_clinicDB, and
Colon_Kvasir; it trades away AUROC on Liver and Colon_colonDB while still
improving AP there. C is inconsistent and loses the aggregate. AC improves
over C, but its gains are insufficient to beat H on both primary metrics.

### Per-dataset Medical results: E20 secondary

Cells are `pixel AUROC / pixel AP`.

| Dataset | H | A | C | AC |
|---|---:|---:|---:|---:|
| Brain | 94.185 / 28.536 | 95.107 / 34.038 | 94.726 / 32.890 | 94.787 / 32.819 |
| Liver | 96.740 / 6.164 | 96.517 / 5.719 | 96.943 / 7.267 | 96.099 / 5.132 |
| Retina | 94.344 / 45.981 | 95.339 / 56.235 | 92.523 / 38.407 | 93.261 / 43.890 |
| Colon_clinicDB | 87.602 / 44.611 | 87.943 / 47.118 | 88.111 / 49.494 | 89.464 / 53.823 |
| Colon_colonDB | 86.765 / 33.578 | 85.227 / 35.328 | 83.979 / 29.630 | 83.589 / 35.190 |
| Colon_Kvasir | 86.358 / 54.971 | 85.815 / 52.851 | 85.543 / 53.574 | 86.759 / 56.938 |

At E20, A - H is `-0.008 / +2.908` in the macro pixel metrics. The AP gain
is concentrated on Brain, Retina, Colon_clinicDB, and Colon_colonDB; A loses
AP on Liver and Colon_Kvasir. E20 is supporting evidence only and does not
override the E15 rule.

## Factorial effects

The factorial contrasts are `AUROC / AP` on the six-dataset Medical pixel
means.

| Horizon | Anchor A-H | CIR C-H | Interaction AC-A-C+H |
|---:|---:|---:|---:|
| E15 | +0.436 / +3.594 | -0.595 / -1.381 | +0.145 / -1.033 |
| E20 | -0.008 / +2.908 | -0.695 / -0.430 | +0.363 / -0.152 |

The interaction is directionally mixed at both horizons, so it is neutral
under the preregistered rule. The frozen decision is `FINAL=A`.

## Comparison with Phase 2

### Macro comparison

| Phase / version | Epoch | Pixel mean 6 AUC / AP | Image mean 3 AUC / AP |
|---|---:|---:|---:|
| Baseline mainbase, pixel-AP best | 9 | 89.77 / 38.24 | 73.33 / 73.89 |
| Phase 1 V3c fp32 anchor | 9 | 90.76 / 39.82 | 73.80 / 75.06 |
| Phase 2B Hybrid alpha0.2 + K-reg 2e-3 | 10 | **90.98 / 40.35** | 73.77 / 74.24 |
| H2 H | E15 | 90.815 / 35.874 | 76.149 / 76.250 |
| H2 A | E15 | **91.252 / 39.468** | 75.283 / 76.343 |
| H2 C | E15 | 90.220 / 34.493 | 75.515 / 75.465 |
| H2 AC | E15 | 90.801 / 37.054 | **76.614 / 76.699** |
| H2 H | E20 | 90.999 / 35.640 | 76.376 / 76.437 |
| H2 A | E20 | 90.991 / 38.548 | 75.788 / 76.501 |
| H2 C | E20 | 90.304 / 35.210 | **76.470 / 76.756** |
| H2 AC | E20 | 90.660 / 37.965 | 76.346 / 76.364 |

The Phase 2B e10 row is the closest Medical comparison, but it is **not a
strict apples-to-apples comparison**: its saved per-dataset output is rounded
to two decimals and uses the older `pixel_stride=4` evaluation path, whereas
H2 uses raw exact metrics with `pixel_stride=1`. Therefore the table is
descriptive, not a new cross-phase significance claim.

### Phase 2B per-dataset comparison

The table below compares the selected Phase 2B e10 row with H2 E15 H and A.
Cells are `pixel AUROC / pixel AP`; the final column is `H2 A15 - Phase2B`.

| Dataset | Phase2B e10 | H2 H15 | H2 A15 | A15 - Phase2B |
|---|---:|---:|---:|---:|
| Brain | 95.24 / 38.28 | 94.683 / 32.076 | 95.459 / 36.196 | +0.219 / -2.084 |
| Liver | 97.07 / 6.81 | 96.712 / 6.109 | 96.380 / 5.467 | -0.690 / -1.343 |
| Retina | 92.20 / 40.39 | 94.686 / 47.596 | 95.360 / 55.069 | +3.160 / +14.679 |
| Colon_clinicDB | 89.06 / 57.46 | 86.982 / 43.964 | 88.657 / 50.466 | -0.403 / -6.994 |
| Colon_colonDB | 83.88 / 35.70 | 86.160 / 32.336 | 85.054 / 35.475 | +1.174 / -0.225 |
| Colon_Kvasir | 88.40 / 63.45 | 85.669 / 53.164 | 86.601 / 54.136 | -1.799 / -9.314 |

The Phase 2B advantage is strongest on Colon_clinicDB and Colon_Kvasir AP,
while H2 A is strongest on Retina AP. The differences should not be
interpreted as a clean architecture win because the evaluator resolution and
rounding differ.

### Phase 2C VisA results

Phase 2C is a separate VisA-only exploratory branch and is not merged into the
Medical comparison.

| Version | Epoch | Pixel AUC / AP | Image AUC / AP | Decision |
|---|---:|---:|---:|---|
| A-prime | 13 | 94.8038 / 55.5341 | 97.9028 / 98.4225 | Selected Phase2C candidate |
| P / full PCGrad | 13 | 97.1696 / 51.7660 | 96.3819 / 96.7979 | Not selected |
| PL / LoRA-only PCGrad | 15 | 96.6840 / 52.7478 | 97.3542 / 97.9956 | Not selected |

PCGrad raises pixel AUROC but reduces pixel AP; A-prime remains the
Pixel-AP-first winner. This is qualitatively consistent with the H2 finding
that a higher AUROC alone is not sufficient for promotion.

## MVTec transfer confirmation

MVTec is a matched industrial transfer check, not a pristine untouched-test
claim. Cells are `pixel AUROC / pixel AP` and `image AUROC / image AP`.

| Arm | Pixel | Image |
|---|---:|---:|
| H15 | 86.869 / 41.612 | 89.526 / 95.207 |
| A15 | 90.041 / 45.159 | 89.817 / 94.779 |
| C15 | 87.210 / 41.041 | 88.255 / 94.749 |
| AC15 | 89.756 / 45.416 | 89.543 / 94.748 |

Relative to H15, A improves pixel AUROC/AP by `+3.173 / +3.547`, while AC
improves pixel AP by `+3.804` but lowers image AP. MVTec does not change the
Medical preregistered selection.

## Authoritative artifacts

- [Published per-dataset comparison CSV](./ACD_CLIP_PHASE_COMPARISON_PUBLISHED.csv)
- [H2 E15 Medical per-dataset raw table](./H2_4ARM_E15_MEDICAL_PER_DATASET.csv)
- [H2 E20 Medical per-dataset raw table](./H2_4ARM_E20_MEDICAL_PER_DATASET.csv)
- [H2 E15/E20 Medical macro summaries](./H2_4ARM_E15_MEDICAL_SUMMARY.csv),
  [E20](./H2_4ARM_E20_MEDICAL_SUMMARY.csv)
- [Factorial effects](./H2_4ARM_FACTORIAL_EFFECTS.csv)
- [Frozen decision](./H2_4ARM_FINAL_DECISION.md)
- [Final protocol audit](../audit/H2_4ARM_FINAL_PROTOCOL_AUDIT.md)
- [E15/E20 freeze manifest](../audit/H2_4ARM_E15_E20_FREEZE.json)
- [MVTec summary](./H2_4ARM_E15_MVTEC_SUMMARY.csv) and
  [per-class table](./H2_4ARM_E15_MVTEC_PER_CLASS.csv)

This is a Git-published reproducibility snapshot on the H2 research branch.
It is not a claim that a paper, release, or upstream pull request has been
accepted by an external venue.
