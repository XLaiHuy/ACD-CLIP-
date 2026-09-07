# H2 S2-LOCR R1 Bounded Decision

* `S2_LOCR_TRAINING_AUTHORIZED=YES`
* `NUMERICAL_VALIDITY=PASS`
* `BOUNDED_SCREEN=FAIL`
* `S2_LOCR_MECHANISM=NOT_SUPPORTED`
* `FULL_CONFIRMATORY_RUN_JUSTIFIED=NO`
* `INTERPRETATION=CASE_F_OR_MECHANISM_GATE_FAILURE`

## Pairing
* attempts: control=500, candidate=500
* successful: control=500, candidate=500
* exact batch identity: `True`
* candidate additional natural skips: `[]`

## Gate results
* `training_authorization=PASS`
* `numerical_validity=PASS`
* `exact_attempted_batch_identity=PASS`
* `successful_step_count_match=PASS`
* `s2_locr_active=PASS`
* `stage2_near_p95_decreases=PASS`
* `stage2_near_p99_decreases=PASS`
* `final_near_p95_not_increased=PASS`
* `final_near_p99_not_increased=PASS`
* `final_ap_non_decrease=PASS`
* `final_auroc_non_decrease=PASS`
* `positive_mean_non_decrease=FAIL`
* `positive_median_non_decrease=FAIL`
* `interior_mean_non_decrease=FAIL`
* `interior_median_non_decrease=FAIL`
* `near_positive_inversion_not_increased=FAIL`
* `near_interior_inversion_not_increased=PASS`

## Interpretation

The result is a source-only exploratory mechanism screen. It does not establish novelty or authorize a confirmatory full run automatically.

Post-run recommendation flag: `NOT_TRIGGERED`.

## Prohibitions and scope
Medical inference: NO; MVTec inference: NO; target tuning: NO; hyperparameter sweep: NO; E15/E20 full training: NO.
WAITING_FOR_USER_APPROVAL=YES
