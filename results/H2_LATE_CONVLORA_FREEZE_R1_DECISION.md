# H2_LATE_CONVLORA_FREEZE_R1 decision

## Decision

- `MECHANISM_TEST=SUPPORTED`
- `BOUNDED_SCREEN=FAIL`
- `LATE_CONVLORA_CAUSAL_HYPOTHESIS=NOT_SUPPORTED`
- `SIMPLE_LATE_STAGE23_FREEZE_NOT_SUPPORTED`
- `SOFT_DRIFT_GOVERNOR_JUSTIFIED=NO`

The screen used the exact E10 Safe-Anchor full state, one matched 500-attempt VisA source stream, and equal inference fusion on the frozen 96-image endpoint cohort. Stage metrics and geometry are reported diagnostics, not hidden gates.

## Direct mechanism gates

- `implementation_parity_pass=True`
- `candidate_stage2_convlora_drift_exactly_zero=True`
- `candidate_stage3_convlora_drift_exactly_zero=True`
- `control_stage2_convlora_drift_gt_zero=True`
- `control_stage3_convlora_drift_gt_zero=True`
- `candidate_stage2_pre_suppression_gradient_meaningful=True`
- `candidate_stage3_pre_suppression_gradient_meaningful=True`
- `non_selected_training_continues=True`

## Scientific gates

- `numerical_validity=True`
- `exact_500_attempted_batch_stream=True`
- `successful_count_match=True`
- `true_freeze=True`
- `final_ap_not_below_control_minus_1e6=False`
- `final_auroc_not_below_control_minus_1e6=True`
- `positive_mean_not_below_control=True`
- `positive_median_not_below_control=True`
- `interior_mean_not_below_control=True`
- `interior_median_not_below_control=True`
- `near_background_p95_not_above_control=False`
- `near_background_p99_not_above_control=False`
- `near_background_gt_positive_inversion_not_above_control=True`
- `near_background_gt_interior_inversion_not_above_control=True`

## Endpoint deltas (candidate minus control)

- AUROC: `0.044960530197924786`
- AP: `-0.0058612052382776`
- Positive mean / median: `0.008315116167068481` / `2.369519461353775e-05`
- Interior mean / median: `0.011896312236785889` / `3.992695565102622e-05`
- Near-background p95 / p99: `0.03233134001493454` / `0.05800473690032959`
- Inversion near-background > positive / interior: `-0.005652784467866101` / `-0.008668872184739884`

No follow-up trust-region/governor run or target evaluation is authorized by this artifact. `WAITING_FOR_USER_APPROVAL=YES`.
