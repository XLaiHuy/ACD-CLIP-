# H2 historical-mixed generalization bottleneck audit

## Decision

`AUDIT_STATUS=PASS`

`BASELINE=HISTORICAL_MIXED_FP16_FP32_V1`

`PRIMARY_BOTTLENECK=LATE_STAGE_FUNCTIONAL_OVERADAPTATION_AND_ANOMALY_UNDERCOVERAGE`

`SECONDARY_BOTTLENECK=SEGMENTATION_GRADIENT_IMBALANCE_AND_CONFLICT`

`TERTIARY_BOTTLENECK=RARE_FP16_GRADIENT_INVALIDITY`

`SELECTED_SOLUTION=STAGE2_STAGE3_FROZEN_E1_FUNCTIONAL_FEATURE_ANCHOR`

The frozen source evidence identifies a late-stage representation and spatial
coverage failure, not unstable DFG routing, small-anomaly specificity, prompt
norm drift, learning-rate schedule failure, or inadequate model capacity.  A
suppresses the negative tail more strongly than H, but it suppresses most
positive interiors much more strongly as well.  Its stage-2/stage-3 features
are far from E1 geometry, and Conv-LoRA alone accounts for 62.58% of A's
parameter-drift energy.  The selected next mechanism is therefore one
training-only functional anchor from frozen E1 features to A's stage-2 and
stage-3 adapted features.  It adds no inference parameters or inference
compute.  It is selected for a future bounded source-only mechanism test; it
was not implemented, tuned, or trained in this audit.

The diagnosis is strong on the complete VisA source population.  Its
contribution to Medical/MVTec transfer remains possible rather than confirmed
because the frozen Seed-0 target artifacts contain aggregate metrics but no
target score maps or features, and Seed-1/2 target evaluation was prohibited
by the replication validity gate.

## Scope and recovery

- The survived H/A inference arrays and CSVs were reused.  Source inference
  was not restarted.
- Both recovery JSON files were valid.  The `.tmp` was a strict superset and
  added A stage 3, so it was selected and promoted.
- Recovery began with 42 unique completed cohorts and computed only the six
  missing A distribution/morphology cohorts.  The final exact manifest has 48
  unique cohorts: 24 H and 24 A.
- Population coverage is all 2,162 VisA test images, 4,324 arm-image rows, and
  25,944 immutable arm-image-stage-branch routing rows.
- No training, parameter update, target inference, target tuning, or
  hyperparameter search was performed.

## Pixel ranking and morphology

### Exact pooled metrics

| map/cohort | H AUROC | A AUROC | A-H | H AP | A AP | A-H |
|---|---:|---:|---:|---:|---:|---:|
| raw pixel map | 0.986025 | 0.966223 | -0.019802 | 0.809829 | 0.510504 | -0.299325 |
| final pixel map | 0.986177 | 0.973198 | -0.012979 | 0.834344 | 0.549835 | -0.284509 |
| image score | 0.978924 | 0.846140 | -0.132784 | 0.984499 | 0.880434 | -0.104064 |
| stage 1 pixel | 0.954213 | 0.954587 | +0.000374 | 0.285074 | 0.352435 | +0.067361 |
| stage 2 pixel | 0.988539 | 0.946898 | -0.041641 | 0.777818 | 0.566280 | -0.211538 |
| stage 3 pixel | 0.974336 | 0.928983 | -0.045353 | 0.614829 | 0.240197 | -0.374632 |

A's final AP is lower in all 12 source categories.  AUROC increases in five,
but those gains do not repair precision ranking.  The stage decomposition
localizes the loss: A improves stage 1 AP but sharply degrades stages 2 and 3.

### Positive coverage rather than background leakage

| distribution statistic | H | A | interpretation |
|---|---:|---:|---|
| positive mean | 0.468384 | 0.150405 | A suppresses anomaly response |
| positive median | 0.410102 | 0.000276 | most A positive pixels receive almost no score |
| interior mean | 0.591176 | 0.169486 | undercoverage is strongest in interiors |
| interior median | 0.788595 | 0.000316 | broad interior collapse |
| boundary median | 0.004951 | 0.000128 | boundaries are also weak |
| negative mean | 0.000186 | 0.000160 | average background is not inflated |
| negative p99 | 0.00000944 | 0.000000526 | A improves the ordinary background tail |
| near-background p99 | 0.923459 | 0.985050 | a narrow hard-edge tail remains |

The score distribution is extremely sparse: A retains a strong extreme
positive tail (positive p99 0.999668) but loses the bulk of anomaly pixels.
That is an anomaly-recall/coverage mechanism, not a global calibration-only
failure.

The macro per-image AP deficit grows from small (-0.01385) to medium
(-0.01945) to large anomalies (-0.04246).  This contradicts a specifically
small-anomaly bottleneck.  Large, interior-rich anomalies are at least as
affected.

## DFG, GAP, SS2D, and local stability

The fixed eight-transform suite covered 96 images and 768 transformed-image
comparisons.  Across transforms, mean map Spearman is 0.94043, mean top-1%
overlap is 0.81961, mean abnormal-routing L1 is 0.001356, and mean feature
cosine is 0.999861.  These satisfy the frozen `HEALTHY` routing gate.

Blur, Gaussian noise, and JPEG are the weakest map perturbations (Spearman
0.88670, 0.86244, and 0.88645 respectively), while their routing changes stay
small.  Thus the output sensitivity is not mediated by unstable DFG weights.

The local frozen-model counterfactual is also insensitive to DFG constants:
changing beta from 0.08 to 0.12 changes subset AP from 0.361343 to 0.361072;
changing tau from 7.2 to 8.8 changes it from 0.361215 to 0.361171.  A small
stage-1 emphasis improves subset AP to 0.364007, whereas stage-2 and stage-3
emphasis reduce it to 0.360222 and 0.359180.  This supports late-stage dilution
but does not justify tuning weights on target data.

Static routing is non-collapsed and branch-sensitive.  A's mean
normal/abnormal routing L1 is 1.20, 0.92, and 1.03 across stages.  The strongest
source morphology association is A stage-2 routing entropy versus anomaly area
(Spearman 0.484; Pearson 0.828), but robustness and local-sensitivity results
show that association is not enough to make DFG instability causal.

`DFG_ROUTING_INSTABILITY=NOT_SUPPORTED`

`DFG_BETA_TAU_BOTTLENECK=NOT_SUPPORTED`

## Feature and prompt geometry

The frozen feature-geometry gate classifies A as `STRONGLY_DEGRADED`: across
all-image regional means, its stage-average feature cosine to E1 is 0.85933
and linear CKA is 0.44250, both below the 0.90/0.85 strong-degradation cutoffs.
H is also strongly degraded (0.89091 cosine, 0.29287 CKA), showing that source
task adaptation broadly leaves the E1 geometry.  A preserves more CKA than H
but still falls far outside the healthy region.  A stage-3 CKA is only 0.18977.

A's source semantic prototype-margin separation is strong at stage 2
(2.0668) but weak at stage 1 (0.0276) and stage 3 (0.6667).  Its main prompts
remain close to the hard branch (mean cosine 0.98827), so wholesale prompt
norm/drift failure is not supported.  A's normal/abnormal prompt cosine is
-0.058 on average, versus -0.730 for H; this is a plausible contributor to the
weak stage-3 semantic separation, but the prompt branch has no intervention
evidence and is not ranked in the Top 3.

`FEATURE_GEOMETRY_STABILITY=STRONGLY_DEGRADED`

`PROMPT_PRIMARY_BOTTLENECK=NOT_SUPPORTED`

## Parameter-family utilization and optimization

At A-E15, parameter-drift energy is concentrated in Conv-LoRA (62.58%), then
image projection (28.88%), DFG Q/K (6.15%), text adapter (1.21%), and SS2D
(1.11%).  Under the frozen rule, Conv-LoRA is `OVERACTIVE`: its drift share
exceeds 60% while functional geometry is degraded.  No other family meets the
overactive rule.

The fixed gradient probe assigns 71.47% of the sum of configured weighted
family norms to image projection and 25.73% to Conv-LoRA.  Across loss terms,
abnormal Dice contributes 79.18% and focal 14.49% of the same descriptive
norm accounting.  DFG Q/K and SS2D contribute only 0.00286% and 0.00003% at
the probed A batches.  They do not meet the formal `UNDERUTILIZED` rule because
their accumulated drift shares exceed 1%, but their instantaneous direct
loss utilization is very low.

The E15 Adam snapshots are finite.  A update-to-weight ratios range from
0.00000499 (SS2D) to 0.000971 (image projection); no family exceeds 10x the A
family median.  Logged learning rates and StepLR decay match the frozen
schedule.  There is no evidence of a learning-rate explosion or corrupt Adam
state.

The historical mixed run still has rare recoverable nonfinite gradients.  In
the discovery A run, 2 of 5,054 attempted E2-E15 batches were skipped
(0.03957%).  In both attempted confirmatory seeds H ended at 5,410 successful
steps and A at 5,411, invalidating H/A target comparisons.  This confirms a
numerical validity blocker, but its contribution to the Seed-0 target gap is
unknown.  BF16 is not selected because the already-frozen matched comparison
was materially worse on both Medical and MVTec.

`LR_BOTTLENECK=NOT_SUPPORTED`

`ADAM_STATE_BOTTLENECK=NOT_SUPPORTED`

`NUMERICAL_VALIDITY_BLOCKER=CONFIRMED`

## Loss-gradient conflict

No loss pair satisfies the frozen severe rule (cosine <= -0.50 in at least two
of three batches).  Systematic moderate conflicts do occur, including:

- classification versus abnormal Dice in Conv-LoRA (mean -0.322; two batches
  <= -0.20);
- classification versus normal Dice in the text adapter (mean -0.254; two
  batches <= -0.20);
- abnormal Dice versus Anchor in DFG Q/K (mean -0.219; two batches <= -0.20);
- classification versus focal in the text adapter (mean -0.390; two batches
  <= -0.20, one below -0.50).

Combined with abnormal-Dice norm dominance, this makes loss geometry a
credible upstream contributor to representation drift.  Without a training
counterfactual it remains `POSSIBLE`, not confirmed.

## Causal bottleneck ranking

The ranking contains exactly one primary, one secondary, and one tertiary
bottleneck.  “Causal status” applies to the named scope; target contribution
is separately bounded.

| rank | bottleneck | causal status | decisive evidence | target-gap scope |
|---|---|---|---|---|
| Primary | late-stage functional overadaptation and anomaly undercoverage | `LIKELY` on source | stage-2/3 AP loss; interior median collapse; strongly degraded feature geometry; Conv-LoRA overactive; stage-weight counterfactual direction | `POSSIBLE`, target maps/features unavailable |
| Secondary | segmentation-gradient imbalance and conflict | `POSSIBLE` | abnormal Dice 79.18% of weighted-norm accounting; systematic active-family conflicts; drift concentrated in the same image families | `POSSIBLE`, no training intervention |
| Tertiary | rare FP16 gradient invalidity | `CONFIRMED` validity blocker | repeatable nonfinite-gradient skips and unequal H/A successful steps in two seeds | target-performance contribution `UNKNOWN` |

Other classifications: routing instability `NOT_SUPPORTED`; beta/tau
sensitivity `NOT_SUPPORTED`; small-anomaly specificity `NOT_SUPPORTED` on
source; LR/Adam failure `NOT_SUPPORTED`; prompt geometry as primary
`NOT_SUPPORTED`; architecture capacity `NOT_SUPPORTED`; target-side feature,
boundary, and morphology attribution `UNKNOWN`.

## Top-3 candidate mechanisms

1. **Selected — stage-2/stage-3 frozen-E1 functional feature anchor.** Add one
   source-training-only penalty that preserves frozen E1 adapted-feature
   geometry at stages 2 and 3.  This directly targets the measured mediator,
   requires no new inference parameters/compute, and is compatible with the
   existing Safe Anchor direction.
2. **Not selected — family-scoped conflict projection.** Remove only the
   component of abnormal-Dice gradients that conflicts with classification in
   Conv-LoRA/image-projection parameters.  Evidence is suggestive but lacks a
   causal training counterfactual.
3. **Not selected — reliability-residual stage fusion.** Retain equal fusion as
   a base and learn a bounded source-only residual toward reliable stages.
   The frozen local counterfactual is favorable, but the effect is small and a
   learned gate adds mechanism complexity.

`SOLUTION_COUNT=3`

`SELECTED_SOLUTION_COUNT=1`

The selected mechanism is a decision, not an authorization to run a full
training job.  Any future test must first be a bounded, preregistered,
source-only mechanism check with no target selection.

## Evidence artifacts

- `audit/H2_MIXED_EXACT_METRIC_PROGRESS.json`
- `audit/H2_MIXED_SOURCE_PIXEL_RANKING.csv`
- `audit/H2_MIXED_SOURCE_PIXEL_RANKING_DERIVED.csv`
- `audit/H2_MIXED_DFG_ROUTING.csv`
- `audit/H2_MIXED_DFG_STABILITY.csv`
- `audit/H2_MIXED_FEATURE_GEOMETRY.csv`
- `audit/H2_MIXED_SEMANTIC_GEOMETRY.csv`
- `audit/H2_MIXED_PROMPT_GEOMETRY.csv`
- `audit/H2_MIXED_ADAPTER_UTILIZATION.csv`
- `audit/H2_MIXED_LOSS_GRADIENT_CONFLICT.csv`
- `audit/H2_OPTIMIZATION_FORENSICS.md`
- `audit/H2_GENERALIZATION_GAP_RECONSTRUCTION.md`

