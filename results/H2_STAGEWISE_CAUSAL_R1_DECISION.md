# H2 Stagewise Causal Localization Audit R1 — Final Decision

* branch: `research/h2-stagewise-causal-localization-audit-r1`
* parent: `f44cca2e163585dba3bbffc99c45518501db4852`
* `S2_LOCR_R1_DECISION_MODIFIED=NO`

## Frozen bottleneck

* `PRIMARY_DIAGNOSIS=CONTEXTUAL_MIXING_WITH_PATCH_AMBIGUITY`
* `CONFIDENCE=MEDIUM`
* `STAGE2_IS_ROOT_CAUSE=YES`

The strongest reviewer-defensible conclusion is a Stage-2-localized but not Stage-2-exclusive failure: a GT-assisted local Stage-2 oracle improves final AP/AUROC on both disjoint cohorts and matched far replacement does not reproduce it, while exact 14x14 patch occupancy shows elevated zero-footprint near scores and large partial-footprint scores. The no-step directional audit localizes suppression to segmentation projection, Conv-LoRA, and image-side mixing, with Conv-LoRA transferring a strong effect into Stage 3. Absolute positive/interior/boundary scores shrink and anomaly-versus-near ranking worsens despite endpoint pixel-ranking gains, so the result is mixed localization and calibration behavior rather than a pure score rescaling claim. Candidate replay parity fails, so update-time trajectory causality is not established.

## Evidence boundaries

* Stagewise unique support: `YES`.
* Patch footprint: `PASS`; diagnosis `PATCH_FOOTPRINT_ALIASING_LIKELY`.
* Calibration-invariant interpretation: `MIXED`.
* Trajectory replay: `FAIL`; candidate parity failed, so no trajectory causality claim is made.

## Conditional R&D

* `RECOMMENDED_NEXT_DIRECTION=UNCERTAINTY_GUIDED_NATIVE_FOOTPRINT_REFINEMENT_HEAD_WITH_STAGEWISE_MARGIN_GATING`
* `IMPLEMENTATION_AUTHORIZED=NO`
* Literature candidates and verified links are in `audit/H2_STAGEWISE_CAUSAL_R1_RESEARCH.md/json`.

## Scope lock

New mechanism training: NO; Medical inference: NO; MVTec inference: NO; target tuning: NO; hyperparameter sweep: NO; original S2-LOCR R1 decision modified: NO; waiting for user approval: YES.
