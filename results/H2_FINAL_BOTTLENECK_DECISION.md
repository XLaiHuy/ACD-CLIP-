# H2 final bottleneck decision

## Required outputs

`PREVIOUS_REPLICATION_PROTOCOL_AUDIT=PASS`

`ANCHOR_REPLICATION_SUPPORT=NOT_CONFIRMED`

`PUBLISHED_PROTOCOL_MATCH=PARTIAL`

`MEDICAL_RESIDUAL_GAP: pixel AUROC +0.436454, pixel AP +3.594117 (Seed-0
E15 discovery A-H only; no valid multi-seed residual gap)`

`MVTEC_RESIDUAL_GAP: pixel AUROC +3.172666, pixel AP +3.547043 (Seed-0 E15
discovery A-H only; no valid multi-seed residual gap)`

`PIXEL_BOTTLENECK=UNRESOLVED_RANKING_LOCALIZATION_CAUSE_NO_SCORE_MAPS`

`LOCALIZATION_FAILURE_MODE=NOT_IDENTIFIED_NO_MASK_JOINED_SCORE_MAPS`

`DOMAIN_SPECIFIC_FAILURE_PATTERN=HETEROGENEOUS_SEED0_PATTERN`

`TRAINING_DYNAMICS_BOTTLENECK=NUMERICAL_VALIDITY_BLOCKER`

`TRAINING_HORIZON_DIAGNOSIS=METRIC_SPECIFIC_E15_E20_SENSITIVITY_NOT_CAUSAL`

`LOSS_BOTTLENECK=NONE_DEMONSTRATED`

`OPTIMIZER_BOTTLENECK=POSSIBLE_BUT_NOT_IDENTIFIED`

`LR_BOTTLENECK=POSSIBLE_BUT_NOT_SUPPORTED`

`INITIALIZATION_SENSITIVITY=NOT_ESTIMABLE`

`ANCHOR_VARIANCE_EFFECT=NOT_ESTIMABLE`

`DOMINANT_DRIFT_FAMILIES=NOT_ESTIMATED_NO_CHECKPOINT_DISTANCE_TABLE`

`GENERALIZATION_BOTTLENECK=UNRESOLVED_WITHOUT_FEATURE_OR_SCORE_MAP_ARTIFACTS`

`MEDICAL_DOMAIN_SHIFT_HYPOTHESIS=UNKNOWN`

`SCORE_BOTTLENECK=UNKNOWN_RANKING_VS_CALIBRATION_NOT_SEPARABLE`

`DFG_BOTTLENECK=POSSIBLE_BUT_NOT_DEMONSTRATED`

`TEXT_BRANCH_BOTTLENECK=POSSIBLE_BUT_NOT_CAUSALLY_IDENTIFIED`

`CAPACITY_BOTTLENECK=UNCERTAIN_BUT_NOT_SUPPORTED`

`HYPERPARAMETER_CAUSE_SUPPORTED=UNKNOWN`

`TOP_ROOT_CAUSE_1=Repeated nonfinite-gradient skips invalidate confirmatory
trajectories`

`TOP_ROOT_CAUSE_1_CONFIDENCE=HIGH_FOR_VALIDITY_ONLY`

`TOP_ROOT_CAUSE_2=Protocol/evaluator incompleteness in external published
comparison`

`TOP_ROOT_CAUSE_2_CONFIDENCE=MODERATE_EXTERNAL_CONFOUND`

`TOP_ROOT_CAUSE_3=Seed and initialization sensitivity cannot be estimated until
valid H/A trajectories exist`

`TOP_ROOT_CAUSE_3_CONFIDENCE=HIGH_UNCERTAINTY_NOT_EFFECT_SIZE`

`ROOT_CAUSE_REDTEAM=PASS`

`REDESIGN_AUTHORIZED=NO`

`FINAL_ACTION=OPTIMIZATION_REPAIR`

`NEXT_FULL_TRAIN_AUTHORIZED=NO`

## Scientific conclusion

Seed-0 provides discovery evidence that A can improve the matched H2 H baseline
on the frozen E15 Medical and MVTec summaries, but Seeds 1 and 2 are invalid
before target evaluation because repeated nonfinite-gradient skips produce
unequal successful optimizer-step counts. The internal protocol/evaluator
contract is documented and passes, while comparison with the public ACD-CLIP
table remains only partial because stride, rounding, and implementation details
are not fully matched. No score maps, rank arrays, feature dumps, or valid
multi-seed target metrics exist to identify a Medical AP, localization,
domain-shift, or MVTec AUROC mechanism. The evidence therefore supports a
source-only numerical-stability investigation and the smallest resulting
optimization repair, not an architecture redesign or a new target experiment.

## Artifact index

- `audit/H2_PUBLISHED_PROTOCOL_FORENSICS.md`
- `audit/H2_OPTIMIZATION_FORENSICS.md`
- `audit/H2_LOSS_GRADIENT_FORENSICS.md`
- `audit/H2_INITIALIZATION_SEED_FORENSICS.md`
- `audit/H2_GENERALIZATION_FORENSICS.md`
- `audit/H2_DFG_PROMPT_FORENSICS.md`
- `audit/H2_RESIDUAL_BOTTLENECK_ROOT_CAUSE_TABLE.md`
- `audit/H2_REDESIGN_GATE.md`
- `results/H2_RESIDUAL_GAP_MASTER_TABLE.csv`
- `results/H2_PIXEL_ERROR_DECOMPOSITION.csv`
- `results/H2_DATASET_CLASS_BOTTLENECK.csv`
- `results/H2_REPRESENTATION_DRIFT.csv`
