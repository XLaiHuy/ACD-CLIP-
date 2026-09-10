# Locked E15 target audit

Status: `COMPLETE_BELOW_TARGET`

The source-only gates locked `hybrid_alpha_max=0.20`, `rho=0.10`, and `TTA-F`
before this replay. This is a post-lock audit of the confirmed E15 checkpoint;
target labels and target metrics were not used for selection or tuning.

| field | value |
|---|---|
| configured protocol | `R3_LOCKED_TTA_F_TARGET_E15_V1` |
| raw evaluator protocol | `R3_LOCKED_TTA_F_TARGET_V1` |
| policy | identity + horizontal flip + vertical flip + horizontal/vertical flip |
| checkpoint | E15 confirmation, epoch 15 |
| checkpoint SHA256 | `c11a73a8c3b20257c0a0f7cfe1328939e7a72f988a3fe3d6644062e167ee9f77` |
| runtime | `21616.22163812793` seconds |
| finite outputs | `true` |
| target selection/tuning | `false` |
| raw JSON SHA256 | `427ccb174c59b6f9141d64dcd90fa2add187616b73a8308a54e9a3484522e944` |

All metrics below are percentages. Medical pixel metrics average all six rows;
Medical image metrics average Brain, Liver, and Retina, the three rows with
both image labels. MVTec metrics average all 15 categories.

| target | Pixel AUROC | Pixel AP | Image AUROC | Image AP |
|---|---:|---:|---:|---:|
| Medical (TTA-F) | 91.336491 | 37.803217 | 75.984017 | 77.713859 |
| MVTec (TTA-F) | 90.289756 | 46.294598 | 89.085496 | 94.614049 |
| Medical no-TTA reference | 91.251786 | 39.468370 | 75.282727 | 76.343364 |
| MVTec no-TTA reference | 90.041289 | 45.159349 | 89.816913 | 94.779362 |

The Medical deltas are `+0.084705/-1.665152/+0.701290/+1.370496` percentage
points for Pixel AUROC/AP and Image AUROC/AP. The MVTec deltas are
`+0.248467/+1.135249/-0.731417/-0.165313` points.

The observed Medical Pixel AP is `37.803217%`, below the `43.03%` target.
Therefore the stop threshold is not met. The next permitted phase is one
source-only selected novelty mechanism, with no target-guided tuning and no
inference-time overhead.

The complete machine-readable evaluator output is in
[`E15_TARGET_EVAL.json`](./E15_TARGET_EVAL.json). The audit summary is in
[`E15_TARGET_AUDIT.json`](./E15_TARGET_AUDIT.json).
