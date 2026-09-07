# H2 Stagewise Causal Localization Audit R1 — Final Decision

* branch: `research/h2-stagewise-causal-localization-audit-r1`
* parent: `f44cca2e163585dba3bbffc99c45518501db4852`
* `S2_LOCR_R1_DECISION_MODIFIED=NO`

## Frozen bottleneck

* `PRIMARY_DIAGNOSIS=CONTEXTUAL_MIXING_WITH_PATCH_AMBIGUITY`
* `CONFIDENCE=MEDIUM`
* `STAGE2_IS_ROOT_CAUSE=NOT_ESTABLISHED`

Measured facts are that the GT-assisted Stage-2 replacement improves pooled endpoint AP/AUROC on both disjoint cohorts, while its anomaly-versus-near AP/AUROC do not improve and matched far-background replacement does not reproduce the endpoint benefit; exact 14x14 occupancy also shows elevated zero-footprint near scores and larger partial-footprint scores at Stage 2. The no-step directional audit finds coupled suppression in segmentation projection and Conv-LoRA, with a strong Stage-3 directional effect for Conv-LoRA, but these are infinitesimal diagnostics rather than updates. The most plausible interpretation is contextual mixing combined with patch-footprint ambiguity and mixed score/localization behavior. Stage 2 is therefore an informative intervention site, not an established sole root cause; attention/SS2D-only causation and update-time compensation remain unsupported because candidate replay parity failed.

## Evidence boundaries

* Stagewise unique support: `NO`.
* Patch footprint: `PASS`; diagnosis `PATCH_FOOTPRINT_ALIASING_LIKELY`.
* Calibration-invariant interpretation: `MIXED`.
* Trajectory replay: `FAIL`; candidate parity failed, so no trajectory causality claim is made.

## Conditional R&D

* `RECOMMENDED_NEXT_DIRECTION=UNCERTAINTY_GUIDED_NATIVE_FOOTPRINT_REFINEMENT_HEAD_WITH_STAGEWISE_MARGIN_GATING`
* `IMPLEMENTATION_AUTHORIZED=NO`
* Literature candidates and verified links are in `audit/H2_STAGEWISE_CAUSAL_R1_RESEARCH.md/json`.

## Scope lock

New mechanism training: NO; Medical inference: NO; MVTec inference: NO; target tuning: NO; hyperparameter sweep: NO; original S2-LOCR R1 decision modified: NO; waiting for user approval: YES.
