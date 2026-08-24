# P25 Gates and Source-Only Policy Contract

## Q1 (all required)

- G1: meaningful `V>1e-12` and `V<-1e-12` support in at least 10/12 classes.
- G2: median held Spearman >= .20.
- G3: positive held Spearman in at least 9/12 classes.
- G4: macro benefit-sign AUROC >= .65.
- G5: macro Benefit-Capture@20 >= .35.
- G6: Benefit-Capture@20 > .20 in at least 9/12 classes.

The unit of transfer is held category, not the 49,152 panel patches.

## Q2 source-only policy

Only after Q1 pass, select among risk quantiles `{.40,.60,.80}` and benefit
quantiles `{.80,.90,.95}`, using `numpy.quantile(method="linear")` on outer
source OOF arrays. ACT iff proposal sign is nonzero, risk <= numeric source
risk threshold, and benefit score > numeric source benefit threshold; otherwise
KEEP. Remove candidates with source wrong-sign >5% or weighted-harm reduction
<50%. Rank the rest by macro source-class normalized target value
`(captured_positive-selected_negative)/max(total_positive,EPS)`, then by more
positive source classes, lower risk quantile, and higher benefit quantile,
with 1e-12 ties. This is calibration only, never an AP estimate.

Q2 requires audit pass, wrong-sign <=5%, weighted-harm reduction >=50%, macro
pAP >= native+.0025, non-regression >=9/12, strict improvement >=7/12,
pAUROC >= native-.005, and macro pAP above frozen harm-only R2-v2 comparator.
No held candidate sweep or held fallback selection occurs.
