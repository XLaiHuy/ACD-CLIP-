# H2 generalization bottleneck V2 decision freeze

This decision is frozen before observing any new candidate endpoint or target
metric.

`PRIMARY_BOTTLENECK=LATE_STAGE_FUNCTIONAL_OVERADAPTATION_AND_ANOMALY_UNDERCOVERAGE`

`PRIMARY_CONFIDENCE=LIKELY_SOURCE_POSSIBLE_TARGET`

`SECONDARY_BOTTLENECK=SEGMENTATION_GRADIENT_IMBALANCE_AND_CONFLICT`

`SECONDARY_CONFIDENCE=POSSIBLE`

`TERTIARY_BOTTLENECK=RARE_FP16_GRADIENT_INVALIDITY`

`TERTIARY_CONFIDENCE=CONFIRMED_VALIDITY_UNKNOWN_TARGET_CONTRIBUTION`

## Evidence

- A loses source stage-2 and stage-3 pixel AP while stage 1 remains healthy;
  positive and interior score coverage collapses, especially at the median.
- A's source feature geometry is strongly degraded relative to E1, and
  Conv-LoRA accounts for 62.58% of A's parameter-drift energy.
- DFG routing is healthy and perturbation-stable; beta/tau sensitivity does
  not support a routing bottleneck.
- No severe loss-gradient conflict meets the preregistered rule. Moderate
  segmentation-related conflicts remain plausible but lack an intervention.
- Repeated recoverable FP16 gradient skips are a confirmed validity blocker,
  not proof of the target generalization gap.

## Rejected or not-primary alternatives

`REJECTED_BOTTLENECKS=DFG_ROUTING_INSTABILITY;DFG_BETA_TAU_SENSITIVITY;SMALL_ANOMALY_SPECIFICITY;LR_OR_ADAM_FAILURE;PROMPT_GEOMETRY_AS_PRIMARY;INSUFFICIENT_CAPACITY`

These alternatives are either contradicted by the fixed source diagnostics or
lack the required evidence. Target-side localization and domain-shift causes
remain unknown because target score maps/features are unavailable and target
evaluation is prohibited in this phase.

## Frozen Top-3 solution table

1. `STAGE2_STAGE3_FROZEN_E1_FUNCTIONAL_FEATURE_ANCHOR` — selected. It
   directly preserves the measured failing mediator, adds no inference
   parameters or inference compute, and has a fixed R1 source-only protocol.
2. `FAMILY_SCOPED_ABNORMAL_DICE_CLASSIFICATION_CONFLICT_PROJECTION` — not
   selected. The gradient evidence is suggestive but not causal.
3. `BOUNDED_RELIABILITY_RESIDUAL_STAGE_FUSION` — not selected. The frozen
   counterfactual is directionally interesting but weaker and more complex.

`SELECTED_MECHANISM=STAGE2_STAGE3_FROZEN_E1_FUNCTIONAL_FEATURE_ANCHOR`

The R1 test may add only the normalized frozen-E1 token-cosine term at stages
2 and 3. It may not add a stage-1 term, second feature loss, architecture
module, lambda sweep, or target-guided selection.

