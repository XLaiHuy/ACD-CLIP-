# Seed-0 export reconciliation

Status: `PASS`.

The original Medical evaluator output stores image and pixel metrics as separate `metric_type` rows. The first compact export joined on dataset alone, which left pixel AUC/AP blank for the three pixel-bearing datasets. The export was corrected to join on `horizon`, `arm`, `dataset`, and `metric_type`, with no inferred values.

The corrected files contain 24 rows for each horizon (four arms x six datasets), preserve complete pixel metrics for Brain, Liver, and Retina, and preserve the three image-only Colon datasets. Direct recomputation from the canonical evaluator rows reproduces all eight arm/horizon macro summaries with maximum absolute difference `0.0` at tolerance `1e-12`.

The machine-readable evidence is in `audit/H2_SEED0_EXPORT_RECONCILIATION.json`.
