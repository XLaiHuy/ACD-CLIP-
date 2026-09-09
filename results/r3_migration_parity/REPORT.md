# R3 migration parity

Status: `PASS`

The canonical A15 checkpoint was evaluated without changing weights. The
Medical replay uses `benchmark_exact`, full-resolution pixels
(`pixel_stride=1`), and the frozen `cls_only` image-score rule. MVTec uses
the same exact pixel evaluator and its established industrial image-score
blend.

| target | Pixel AUROC | Pixel AP | Image AUROC | Image AP | migration result |
|---|---:|---:|---:|---:|---|
| Medical (six-pixel / three-image macro) | 91.252837 | 39.468876 | 75.282542 | 76.343149 | GREEN |
| MVTec (15-class macro) | 90.041286 | 45.159330 | 89.816913 | 94.779362 | GREEN |
| H2 A15 reference | 91.251786 | 39.468370 | 75.282727 | 76.343364 | — |
| H2 A15 MVTec reference | 90.041289 | 45.159349 | 89.816913 | 94.779362 | — |

All metric drifts are below `0.30` percentage points; the largest is the
display-precision Medical pixel AUROC drift of `0.00105` points. Checkpoint,
CLIP, and dataset manifest SHA256 values are recorded in
[`provenance.json`](./provenance.json). Dataset extraction and structural
validation are recorded in [`R3_DATA_PROVISIONING.md`](/workspace/R3_DATA_PROVISIONING.md).

The first Medical replay intentionally exposed a protocol mismatch: the
pre-fix evaluator blended classification and pixel-peak image scores, while
the frozen H2 contract is `cls_only`. That diagnostic run is retained under
`/tmp/r3_migration_parity_20260909`; only the corrected replay under
`/tmp/r3_migration_parity_20260909_cls_only` is used above.
