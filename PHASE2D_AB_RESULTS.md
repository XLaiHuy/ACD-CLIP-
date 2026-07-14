# Phase2D A-prime/B Interpolation Results

## Parent reproduction

The parent reproduction gate passed within 0.05 percentage points for every registered macro metric.

## Candidate macro metrics

| Candidate | Pixel AUC | Pixel AP | Image AUC | Image AP |
| --- | ---: | ---: | ---: | ---: |
| AB25 | 95.5693 | 55.4652 | 97.9653 | 98.4647 |
| AB50 | 96.0225 | 55.4166 | 97.8611 | 98.3772 |
| AB75 | 96.1600 | 55.2958 | 97.9792 | 98.5016 |

## Decision

A-prime remains the primary winner. The secondary Pareto candidates are exploratory; the locked three-point interpolation test is closed. A later LB_0p1 preregistration may be considered.

This is a single seed-42 comparison and does not establish statistical robustness.

Per-category metrics are retained in `runs/phase2d_ab_interpolation_seed42/visa_val_metrics.csv`.

## Reproduction deltas

| Parent | Pixel AUC delta | Pixel AP delta | Image AUC delta | Image AP delta |
| --- | ---: | ---: | ---: | ---: |
| A-prime | -0.0001 | -0.0000 | -0.0000 | -0.0000 |
| B | +0.0000 | -0.0001 | +0.0000 | +0.0000 |

## Candidate deltas

| Candidate | Pixel AUC vs A-prime | Pixel AP vs A-prime | Image AUC vs A-prime | Image AP vs A-prime | Pixel AP vs B |
| --- | ---: | ---: | ---: | ---: | ---: |
| AB25 | +0.7655 | -0.0689 | +0.0625 | +0.0422 | +0.3310 |
| AB50 | +1.2187 | -0.1175 | -0.0417 | -0.0453 | +0.2824 |
| AB75 | +1.3562 | -0.2383 | +0.0764 | +0.0791 | +0.1616 |

## Per-category summary

Each candidate retained metrics for all 12 VisA categories. The highest category Pixel AP was cashew for AB25 (95.0784) and AB50 (95.2379), and pipe_fryum for AB75 (95.2855). The lowest category Pixel AP was macaroni2 for AB25 (7.2542), AB50 (8.1656), and AB75 (8.8516). Full per-category rows are in the registered CSV artifact.
