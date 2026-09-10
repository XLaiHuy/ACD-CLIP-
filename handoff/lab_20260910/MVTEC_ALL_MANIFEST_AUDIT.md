# MVTec All Supervised Manifest Audit

Source root: `/workspace/datasets/mvtec-ad`
Manifest: `dataset/hub/MVTec_all_supervised.jsonl`

## Verification

- Total images: 5354
- Normal count: 4096
- Anomaly count: 1258
- Train/good count: 3629
- Test/good count: 467
- Test/anomaly count: 1258
- Missing file count: 0
- Missing mask count: 0
- Duplicate count: 0
- Manifest SHA256: `1b850c1b52de3a08db0ad4e14adb51933f716bb69e66776a9ed5cde71f6d7129`

## Per-class counts

| Class | Train/good | Test/good | Test/anomaly | Total |
|---|---:|---:|---:|---:|
| bottle | 209 | 20 | 63 | 292 |
| cable | 224 | 58 | 92 | 374 |
| capsule | 219 | 23 | 109 | 351 |
| carpet | 280 | 28 | 89 | 397 |
| grid | 264 | 21 | 57 | 342 |
| hazelnut | 391 | 40 | 70 | 501 |
| leather | 245 | 32 | 92 | 369 |
| metal_nut | 220 | 22 | 93 | 335 |
| pill | 267 | 26 | 141 | 434 |
| screw | 320 | 41 | 119 | 480 |
| tile | 230 | 33 | 84 | 347 |
| toothbrush | 60 | 12 | 30 | 102 |
| transistor | 213 | 60 | 40 | 313 |
| wood | 247 | 19 | 60 | 326 |
| zipper | 240 | 32 | 119 | 391 |

The manifest uses official MVTec ground-truth masks for every test anomaly. Normal rows intentionally omit `mask_path`.
