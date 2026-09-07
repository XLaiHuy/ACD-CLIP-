# H2 Boundary Spillover R1 Decision

Audit-only conclusion

* **PRIMARY_DIAGNOSIS:** `STAGE_LOCAL_SPILLOVER_LIKELY`
* **SPILLOVER_CONFIDENCE:** `HIGH`
* **MOST_RESPONSIBLE_STAGE:** `STAGE2`
* **INTERPOLATION_CONTRIBUTION:** `MINOR`
* **FUSION_CONTRIBUTION:** `NONE`
* **BOUNDARY_ANNOTATION_ARTIFACT_DOMINANT:** `False`

## Authorized next-step flags

* **LOCAL_BOUNDARY_CONTRAST_JUSTIFIED:** `YES`
* **INFERENCE_RESAMPLING_DIAGNOSTIC_JUSTIFIED:** `NO`
* **FUSION_DIAGNOSTIC_JUSTIFIED:** `NO`
* **RETURN_TO_REPRESENTATION_DIAGNOSIS:** `NO`

The decision is descriptive and uses the fixed cohort, fixed existing mask semantics, fixed current score path, and fixed endpoint-only comparisons. No intervention was implemented automatically. User approval is required before any future Local Boundary Contrast or other diagnostic is run.

## Primary endpoint headline

Safe-Anchor E10 final near-background p95=0.05085538029670707, p99=0.3168737697601336; far-background p95=1.538422189639732e-08, p99=5.3546010683192163e-08; final distance Spearman=-0.3426570446465868; image leakage fraction=0.6875; category majority-rule fraction=1.0.

## Prohibitions

TRAINING_EXECUTED=NO; OPTIMIZER_STEP_EXECUTED=NO; BACKWARD_EXECUTED=NO; NEW_LOSS_OR_REGULARIZER=NO; MEDICAL_EVALUATION_EXECUTED=NO; MVTec_EVALUATION_EXECUTED=NO; TARGET_INFERENCE_OR_TUNING=NO; EPOCH_SELECTION_OR_SWEEP=NO; MORPHOLOGY_RADIUS_SWEEP=NO; THRESHOLD_OR_TOPK_SWEEP=NO; AUTOMATIC_INTERVENTION_EXECUTED=NO.

WAITING_FOR_USER_APPROVAL=YES
