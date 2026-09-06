# H2 master research decision V2

`RECOVERY_STATUS=PARTIAL`

`THREE_LAYER_AUDIT=TIER1_PASS;TIER2_PARTIAL;TIER3_PARTIAL`

`PRIMARY_BOTTLENECK=LATE_STAGE_FUNCTIONAL_OVERADAPTATION_AND_ANOMALY_UNDERCOVERAGE`

`SECONDARY_BOTTLENECK=SEGMENTATION_GRADIENT_IMBALANCE_AND_CONFLICT`

`TERTIARY_BOTTLENECK=RARE_FP16_GRADIENT_INVALIDITY`

`EVIDENCE_STRENGTH=PRIMARY_LIKELY_SOURCE_POSSIBLE_TARGET;SECONDARY_POSSIBLE;TERTIARY_CONFIRMED_VALIDITY_ONLY`

## Top-3 solutions

1. Stage-2/stage-3 frozen-E1 functional feature anchor — selected for the
   bounded test, but R1 mechanism screen failed.
2. Family-scoped abnormal-Dice/classification conflict projection — not run;
   evidence remains only possible.
3. Bounded reliability-residual stage fusion — not run; weaker and more
   complex than the selected direct intervention.

`SELECTED_SOLUTION=STAGE2_STAGE3_FROZEN_E1_FUNCTIONAL_FEATURE_ANCHOR`

## Mechanism decision

The R1 test reused the preregistered fixed lambda `37.42480105109332`, fixed
500-attempt horizon, shared E1, matched augmentation stream, historical mixed
FP16/FP32 precision, Safe Anchor, DFG, SS2D, optimizer, scheduler, and loss.
The candidate improved stage-2 and stage-3 cosine-to-E1 geometry on the fixed
96-image VisA source subset, but reduced final source AUROC/AP and positive
and interior coverage. The control had two recoverable gradient skips, so
the numerical gate also failed. This is `BOUNDED_SCREEN_RESULT=FAIL` and
`MECHANISM_SUPPORTED=NO`.

The frozen bottleneck is not rejected: the intended mediator moved in the
predicted direction, but this particular regularizer damages the measured
ranking/coverage endpoint at the authorized strength and horizon. No tuning
is allowed after this result.

`WHY_NOT_ALTERNATIVES=TOP2 lacks likely-or-confirmed causal support; TOP3 is
less direct and more complex; no candidate is authorized after a failed R1
screen without new diagnostic evidence.`

`PUBLISHED_COMPARATOR_STATUS=PARTIAL_PROTOCOL_MATCH_CONTEXT_ONLY`

Published ACD-CLIP values remain contextual only. No target metrics were used
in this decision, and no claim of matched superiority is made.

`TARGET_TUNING_USED=NO`

`NEXT_FULL_E15_JUSTIFIED=NO`

`FULL_E15_STARTED=NO`

`MEDICAL_INFERENCE_RUN=NO`

`MVTEC_INFERENCE_RUN=NO`

