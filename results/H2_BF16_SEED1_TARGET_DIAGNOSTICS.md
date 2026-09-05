# H2 BF16 Seed1 target diagnostics

This is a descriptive analysis of the frozen H/A E15 target outputs. It does not change checkpoints, rerun inference, select a model, or select a target-dependent setting.

| Artifact | Medical | MVTec AD | Status |
|---|---|---|---|
| Raw image scores with labels and GT file joins | saved | not saved | Medical supported |
| Image ROC/PR arrays | derived from saved raw scores | unavailable | saved in `H2_BF16_SEED1_MEDICAL_IMAGE_CURVES.json` |
| Positive/negative image-score distributions | derived | unavailable | saved in `H2_BF16_SEED1_MEDICAL_IMAGE_SCORE_DISTRIBUTIONS.csv` |
| Top image-level rank errors | derived | unavailable | saved in `H2_BF16_SEED1_MEDICAL_IMAGE_RANK_ERRORS.csv` |
| Raw pixel scores, score maps, and pixel GT joins | not persisted | not persisted | unsupported by the frozen evaluators |
| Pixel PR/ROC arrays, per-image pixel AP, boundary/interior analysis | unavailable | unavailable | unsupported without a new evaluator run |
| Target DFG weights, prompt/prototype cosines, and feature statistics | not emitted | not emitted | unsupported by the frozen evaluators |

The Medical evaluator retained 9,005 image predictions per arm across six datasets. ROC/PR curves are exact rank curves of the saved `cls_score` and labels. They are defined only for Brain, Liver, and Retina, which each contain both image classes; the three Colon splits have anomaly-only image labels and therefore do not have defined image ROC/PR curves.

The rank-error file lists the ten highest-scoring negatives and ten lowest-scoring positives per dataset and arm, solely as descriptive ranking evidence. It does not choose a threshold or motivate a target-dependent intervention.

The primary outcomes are pixel metrics. Because neither evaluator retained pixel arrays/maps, the current data cannot distinguish pixel ranking, calibration, boundary localization, or small-anomaly sensitivity. This is an observability limitation, not evidence for a model or architecture change. No additional target evaluation was run to fill the gap.
