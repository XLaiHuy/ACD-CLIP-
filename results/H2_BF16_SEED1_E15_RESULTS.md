# H2 BF16 Seed1 E15 results

Fresh matched Seed1 BF16 source training completed for H and A before target
access. Both E15 checkpoints are numerically valid: 5,415 successful steps,
zero nonfinite loss/gradient events, finite parameters, and finite Adam state.
Checkpoint identities and numerical evidence are in
`audit/H2_BF16_SEED1_E15_NUMERICAL_AUDIT.md`.

| target evaluator | arm | pixel AUROC | pixel AP | image AUROC | image AP |
|---|---|---:|---:|---:|---:|
| Medical raw exact (6 pixel / 3 image datasets) | H | 87.1502 | 30.5611 | 75.6127 | 76.8170 |
| Medical raw exact (6 pixel / 3 image datasets) | A | 88.8694 | 34.6473 | 75.8583 | 77.0199 |
| MVTec benchmark exact (15 classes) | H | 84.1344 | 40.4316 | 91.1364 | 95.7989 |
| MVTec benchmark exact (15 classes) | A | 84.8983 | 41.8373 | 89.9589 | 95.1743 |

For Medical, A minus H is +1.7192 pixel AUROC, +4.0862 pixel AP, +0.2456
image AUROC, and +0.2029 image AP. For MVTec, A minus H is +0.7639 pixel
AUROC, +1.4057 pixel AP, -1.1775 image AUROC, and -0.6246 image AP.

These are **Seed1-only BF16** outcomes under a changed precision policy. They
are not a matched comparator for the historical FP16 Seed0 evidence and must
not be pooled with it. No C/AC arm, no Seed2 BF16 arm, no hyperparameter
tuning, and no further training was performed.

## Frozen artifacts and descriptive diagnostics

- [E15 checkpoint manifest](./H2_BF16_SEED1_E15_MANIFEST.json)
- [per-dataset/per-class target metrics](./H2_BF16_SEED1_TARGET_PER_DATASET.csv)
- [master comparison](./H2_BF16_SEED1_MASTER_COMPARISON.md)
- [target-output diagnostic inventory](./H2_BF16_SEED1_TARGET_DIAGNOSTICS.md)
- [final bottleneck decision](../audit/H2_BF16_SEED1_FINAL_BOTTLENECK_DECISION.md)
