# Phase2C BF16 A-prime/B diagnostic

Selected epochs: A-prime e13; B e13.

## Macro delta (B - A-prime)

| Pixel AUC | Pixel AP | Image AUC | Image AP |
|---:|---:|---:|---:|
| 1.4198 | -0.4000 | -0.0278 | 0.0062 |

## Decision gate

- Shared-path conflict flag: `True` (negative cosine rate > 0.50).
- Norm-imbalance flag: `True` (median CLS/SEG norm ratio > 10.0).
- Recommended branch: `DIAGNOSIS_REQUIRED`.

Threshold-based flags are screening signals. Pre-register thresholds before treating the branch choice as confirmatory.
