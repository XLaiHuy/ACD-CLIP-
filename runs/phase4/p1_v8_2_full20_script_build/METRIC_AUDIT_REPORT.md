# Pre-Train/Test Metric Audit Report

**Repository**: `/home/ai4/caohuy/ACD-CLIP-phase4`  
**Branch**: `phase4-progress1-cops-dynamic-prompt`  
**Commit**: `96c5b9c6ad8ec2b3b2eaec11a5b0deab58d41b2c`  
**Date**: 2026-08-06  
**Metric Audit Decision**: `METRICS_READY`

---

## 1. Metric Semantics & Definitions

- **Image Anomaly Score**: Combined classification logit / probability score:
  $$\text{pred\_image} = 0.9 \times P(\text{abnormal}) + 0.1 \times \max(\text{pixel\_pred})$$
  (or $0.5 / 0.5$ for Medical domain), where $P(\text{abnormal}) = \text{Softmax}(\text{cls\_preds})[:, 1]$.
- **Pixel Anomaly Score**: Continuous patch fusion score map `seg_pred` output by `vision_text_fusion_gate_seg`.
- **Normal Class Index**: `0`
- **Anomaly Class Index**: `1`
- **Score Direction**: Larger values strictly mean "more anomalous".

---

## 2. Image-Level Metric Requirements

- **AUROC & AP**: Evaluated globally over the entire test set.
- All batch-level image predictions and labels are collected into dataset-level tensors before computing `auroc` and `average_precision`.
- **Single-class failure handling**: Single class in target labels explicitly raises `ValueError`.

---

## 3. Pixel-Level Metric Requirements

- Predictions are resized to ground-truth mask resolution before flattening.
- Interpolation: Bilinear for continuous predictions, Nearest for ground-truth masks.
- Masks are binary $\{0, 1\}$.
- AUROC & AP are computed globally over all test pixels combined (not averaged per image).

---

## 4. Multi-Level Aggregation Order

1. Per-level base logits calculated.
2. Rho-scaled H6 correction applied to abnormal channel:
   $$\text{corrected\_logits}[..., 1] = \text{base\_logits}[..., 1] + \rho \cdot \text{h6\_logits}$$
3. Interpolated across spatial grid.
4. Aggregated across groups.
5. Softmax applied once to produce probabilities.

---

## 5. Test Loader & Accounting Safeguards

- `stage = "test"` (official test split).
- `shuffle = False`.
- `drop_last = False`.
- Evaluated sample count must equal expected official test set count.

---

## 6. Synthetic Unit Test Results (`tests/test_p1_v8_2_metrics.py`)

- `test_image_auroc_perfect`: PASSED
- `test_image_ap_perfect`: PASSED
- `test_image_metrics_reverse_scores`: PASSED
- `test_image_metric_length_mismatch_fails`: PASSED
- `test_image_metric_single_class_fails_explicitly`: PASSED
- `test_pixel_auroc_perfect`: PASSED
- `test_pixel_ap_perfect`: PASSED
- `test_pixel_global_not_mean_per_image`: PASSED
- `test_prediction_mask_shape_parity`: PASSED
- `test_mask_binary_check`: PASSED
- `test_nonfinite_scores_fail`: PASSED
- `test_test_loader_drop_last_false`: PASSED
- `test_test_loader_shuffle_false`: PASSED
- `test_full_sample_accounting`: PASSED
- `test_abnormal_score_direction`: PASSED
- `test_no_double_softmax`: PASSED
- `test_epoch_result_schema`: PASSED

**17 passed in 2.57s.**
