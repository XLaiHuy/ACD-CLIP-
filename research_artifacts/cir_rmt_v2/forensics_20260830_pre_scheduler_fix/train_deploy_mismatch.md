# Train/deploy mismatch audit

Status: PASS

This is a read-only deterministic first-batch audit at the preserved V2 E14 checkpoint. The training-side map intentionally uses the production training probability path; the deployment-side map intentionally uses the frozen deployment path.

| quantity | value |
|---|---:|
| mean_abs_difference | 0.004018271807581186 |
| max_abs_difference | 0.9989231824874878 |
| pearson_correlation | 0.6329717274242498 |
| training_pixel_auroc | 0.9976063522365747 |
| training_pixel_ap | 0.6854987757102292 |
| deployment_pixel_auroc | 0.995797234447703 |
| deployment_pixel_ap | 0.5536414430731527 |
| classification_loss | 0.3635416328907013 |
| segmentation_loss | 0.36348167061805725 |
| weighted_kg_loss | 0.0003362131246831268 |
| weighted_k_loss | 0.0 |

Interpretation: a nonzero map difference is proven deployment/training-path divergence. Its effect on the final six-medical failure is correlational until matched full-evaluation evidence isolates it.

Evidence: `tools/cir_rmt/runtime.py:60-70`, `model/phase2b_runtime.py:164-184`, and `gradient_conflict_report.csv` in this directory.
