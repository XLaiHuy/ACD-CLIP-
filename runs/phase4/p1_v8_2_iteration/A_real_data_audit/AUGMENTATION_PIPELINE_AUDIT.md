# Augmentation Pipeline Audit

## Exact Source Paths
- `dataset/__init__.py` (class `TextAndImageDataset`)

## Implementation Details
- **Image Loading**: `Image.open().convert("RGB")`
- **Mask Loading**: `Image.open().convert("L")`
- **Initial Resize**: Image uses `InterpolationMode.BICUBIC`, Mask uses `InterpolationMode.NEAREST`.
- **Normalization**: Image is normalized with CLIP mean `(0.48145466, 0.4578275, 0.40821073)` and std `(0.26862954, 0.26130258, 0.27577711)` *before* joint spatial transformations.
- **Joint Spatial Transformations**:
  - `RandomRotation(degrees=30)`, p=0.5
  - `RandomAffine(degrees=0, translate=(0.15, 0.15))`, p=0.5
  - `RandomHorizontalFlip(p=0.5)`
  - `RandomVerticalFlip(p=0.5)`
- **Image Label**: Sourced from `meta["label"]`.
- **Mask Output**: Sourced from `(mask != 0).float()`.

## Deterministic Probe Findings
- **Shared Geometric Parameters**: Yes, image and mask are concatenated into a 4-channel tensor (`torch.cat([img, mask], dim=0)`) prior to spatial transforms.
- **Interpolation Mode (Forced)**: Because image and mask are concatenated into a tensor, `RandomRotation` and `RandomAffine` apply `InterpolationMode.NEAREST` (the torchvision default for tensors) to *both* the image and the mask. This preserves the binary nature of the mask but degrades image quality during augmentation (nearest neighbor rotation/translation instead of bilinear/bicubic).
- **Fill Values**: The default `fill=0.0` is used for both image and mask. 
  - For the mask, 0.0 correctly represents background.
  - For the image, because normalization is applied *before* the spatial transform, a fill value of 0.0 in the normalized tensor corresponds to the mean pixel color in the original image space (`(0.0 * std) + mean = mean`). The padding is effectively a mid-gray-brown color rather than black or zero in pixel space.
- **Mask dtype/range**: The mask is a `float32` tensor containing values exactly `0.0` and `1.0`.

## Conclusion
The joint augmentation pipeline is deterministic and properly aligns masks with images. However, it forces nearest-neighbor interpolation on images and uses mean-color padding. This is noted but does not strictly invalidate the audit or training.
