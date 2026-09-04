# H2 generalization and domain-shift forensics

## Evidence boundary

Only the frozen Seed-0 E15 target results are available for H/A comparison.
Seed-1 and Seed-2 target evaluation was correctly not run after hard training
invalidity. The current results are therefore discovery evidence, not robust
multi-seed generalization evidence.

## Medical per-dataset pattern

At E15, A versus H wins both pixel AUROC and AP on Brain, Retina,
`Colon_clinicDB`, and `Colon_Kvasir`. It regresses on both metrics on Liver.
On `Colon_colonDB`, AUROC decreases by `1.105664` points while AP increases by
`3.139545` points. The six-dataset macro contrast is:

- pixel AUROC: `+0.436454` points;
- pixel AP: `+3.594117` points.

This is heterogeneous behavior, not a uniform Medical effect.

## MVTec per-class pattern

The 15-class table in `results/H2_DATASET_CLASS_BOTTLENECK.csv` preserves all
classes, including negative cases. A wins both metrics on 10 classes, regresses
on both on cable, leather, and pill, improves AUROC while reducing AP on
zipper, and reduces AUROC while improving AP on toothbrush. The Seed-0 macro
contrast is pixel AUROC `+3.172666` and AP `+3.547043` points.

Using the fixed descriptive class-name grouping already encoded in the CSV,
the five texture classes average `+2.693448` AUROC and `+1.280342` AP, while
the ten object classes average `+3.412274` AUROC and `+4.680393` AP. This
descriptive split is not a tuned cluster and is not a causal claim. The
negative cable/leather/pill cases demonstrate why the macro gain must not be
presented as universal.

## External comparison

The public N=3 Medical values are close in some AUROC rows but AP differs
substantially for several datasets. The public MVTec macro is `91.4/43.6`
(pixel AUROC/AP), compared with current A `90.041289/45.159349`. Because the
public protocol is incomplete and the current exact evaluator differs from the
historical rounded/stride-4 evaluator, these are approximate context only.

## Missing representation and morphology evidence

No target score maps, per-image mask joins, PR/ROC arrays, feature dumps, CKA,
prototype distances, covariance shifts, or anomaly-size stratifications are
archived. The required diagnostic tables explicitly mark those quantities as
unavailable. Thus the following are unresolved rather than silently inferred:

- whether Medical AP is driven by false-positive tails, weak peaks, boundary
  errors, or small lesions;
- whether Medical has a larger feature-domain shift than MVTec;
- whether A retains more transferable CLIP geometry than H;
- whether regressions are texture-, object-, size-, or morphology-driven.

`DOMAIN_SPECIFIC_FAILURE_PATTERN=HETEROGENEOUS_SEED0_DISCOVERY_PATTERN`.
`GENERALIZATION_BOTTLENECK=UNRESOLVED_WITHOUT_FEATURE_OR_SCORE_MAP_ARTIFACTS`.
`MEDICAL_DOMAIN_SHIFT_HYPOTHESIS=UNKNOWN`.
