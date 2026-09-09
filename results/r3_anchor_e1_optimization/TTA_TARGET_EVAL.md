# Locked target TTA-F evaluation

Status: `PASS`

The source-only gate locked `TTA-F` before this replay. This evaluation uses
the canonical A15 checkpoint without changing weights or selecting anything
from target labels.

| field | value |
|---|---|
| protocol | `R3_LOCKED_TTA_F_TARGET_V1` |
| policy | identity + horizontal flip + vertical flip + horizontal/vertical flip |
| checkpoint | canonical A15, epoch 15 |
| checkpoint SHA256 | `727dc4813db1ef0c5a6db3a4cf15e916ad1413dcf629913358419d2734edecfe` |
| runtime | `18787.844135929015` seconds |
| finite outputs | `true` |
| target selection/tuning | `false` |
| raw JSON SHA256 | `3ab746654fb1e5599cec63a994e4a5d4c35d953619720b67801004fc3229dcb3` |

All target metrics below are percentages. Medical image metrics are averaged
over the three image-bearing rows (Brain, Liver, Retina), matching the
migration report; Medical pixel metrics use all six rows. Image-only Colon
rows have no image AUROC/AP and are excluded from that image macro.

| target | Pixel AUROC | Pixel AP | Image AUROC | Image AP |
|---|---:|---:|---:|---:|
| Medical (TTA-F) | 91.755123 | 40.245920 | 74.683768 | 76.700503 |
| MVTec (TTA-F) | 90.416296 | 47.297504 | 90.776911 | 95.272454 |
| Medical no-TTA reference | 91.251786 | 39.468370 | 75.282727 | 76.343364 |
| MVTec no-TTA reference | 90.041289 | 45.159349 | 89.816913 | 94.779362 |

The corresponding target replay deltas are `+0.503337/+0.777551/-0.598959/+0.357139`
percentage points for Medical Pixel AUROC/AP and Image AUROC/AP, and
`+0.375007/+2.138155/+0.959998/+0.493092` points for MVTec. These are
post-lock diagnostic results and do not alter the source-selected TTA policy.

The complete machine-readable result is in
[`TTA_TARGET_EVAL.json`](./TTA_TARGET_EVAL.json). The run was executed from
the locked A15 checkpoint with batch size 16 and two workers; the intermediate
spools were removed by the exact accumulator after each class.
