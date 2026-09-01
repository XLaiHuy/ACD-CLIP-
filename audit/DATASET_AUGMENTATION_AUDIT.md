# Dataset and augmentation audit

## Manifest integrity

The audit was run against the candidate checkout at the exact H2 source
commit and the canonical data root `/home/ai4/caohuy/data`. Every referenced
image and mask exists; image and mask paths are unique within each manifest;
normal rows have no mask and anomaly rows have a mask.

| manifest | rows | labels | SHA256 | missing image/mask |
|---|---:|---|---|---:|
| VisA | 2162 | 0:962, 1:1200 | `468463d2d6234fa7537c6da32b027758527676a12a54a4028c5a282cdd726842` | 0/0 |
| Brain | 3715 | 0:640, 1:3075 | `89092dd5f3e36d2e611b115b2a97e4e9ee83af183ebec298abac983a7a323e4e` | 0/0 |
| Liver | 1493 | 0:833, 1:660 | `1483b5a43f011a3ef02211d5fa81c5b09031423bd0ca5c0ef6cbf0375fee4fc8` | 0/0 |
| Retina | 1805 | 0:1041, 1:764 | `d0de975045262b321851ac3770eb7b5e68d4d7fb3bdba833b1cbbbe32f212e24` | 0/0 |
| Colon_clinicDB | 612 | 1:612 | `1f057657a64221672a5123c3e87b926d226b9eb6a3276768385ca3a7554cdb5c` | 0/0 |
| Colon_colonDB | 380 | 1:380 | `e3be9a5e158bef9a2c7f481827339798152c78e92590df9631c4281a6b6c31c3` | 0/0 |
| Colon_Kvasir | 1000 | 1:1000 | `ac948309511f02e8ec66b9c3b5dbc4a4be5e85d22dbd17742c74212ceee2ee94` | 0/0 |

All seven manifests have one class per evaluator dataset, zero duplicate
image paths, zero duplicate mask paths, and zero anomaly rows without masks.

## Historical training path

`TextAndImageDataset` reads `dataset/hub/VisA.jsonl` as the H2 training pool.
The manifest has no split field, so this exact source contract is not a
train/validation/generalization split. The historical 96-image source gate
uses the same trained VisA categories and is therefore an assessment sample,
not an unseen-category counterfactual. The clean factorial config disables
that gate and records the reason explicitly.

The following legacy behavior is intentionally retained:

- Image noise is `AddGaussianNoise(std=1, p=0.7)` whose implementation returns
  the input when the random draw is below `p`; noise is therefore applied with
  probability 0.3. This is recorded as `LEGACY_P_INVERSION`, not silently
  changed.
- Color jitter uses independent `RandomApply(..., p=0.7)` operations.
- Rotation, translation, horizontal flip, and vertical flip are applied to
  the concatenated image-plus-mask tensor, preserving image/mask geometry.
- Training images use bicubic resize and normalization; masks use nearest
  resize, are binarized before geometry transforms, and normal masks are
  zero-filled.
- Medical evaluation uses deterministic resize/normalization and nearest
  mask resize.

The all-positive colon manifests intentionally produce zero image AUROC/AP in
the evaluator's single-class guard; this is a dataset property, not a newly
introduced model failure.

**Status:** manifest file integrity `PASS`; source-gate protocol
`ISSUES_SOURCE_GATE_OVERLAP` and therefore not authorized for target selection.
