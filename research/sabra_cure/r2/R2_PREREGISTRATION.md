# SABRA-CURE R2 Preregistration v1

Status: `FROZEN_BEFORE_IMPLEMENTATION_AND_RESULTS`

Base/P8 terminal SHA: `1fa8775367d4139580ad0abd5a1ed48a96edeb43`.

## Question and single attempt

Can the frozen R1 signed direction score identify source-side interventions
safe enough to apply at fixed alpha `0.25`, while abstaining to native KEEP
otherwise? There is exactly one 12-fold VisA outer-LOCO attempt. R2 does not
learn correction magnitude; it does not run R3/R4, MVTec, or Medical.

## Frozen inputs and model

Inputs are only the immutable R1 source/Trust/R0 caches and the exact 14
GT-free features, in R1 order. Direction fitting exactly reuses R1: train-only
median/IQR (linear NumPy quantiles; IQR floor `1e-6`), float64 centered closed
form ridge (`lambda=1.0`, unregularized intercept, `numpy.linalg.solve`), and
`y=tanh(u/P75(abs(u_train)))` with the training-fold scale floor `1e-8`.

For outer holdout H, each class in the remaining eleven receives an inner-LOCO
direction prediction from a model that excludes that class. Only those
inner-OOF residuals define `z=log(abs(y_cf-mu_cf)+1e-4)`. The uncertainty ridge
is fit on outer-training standardized GT-free features and z; it predicts
`sigma=exp(clip(z_hat,log(1e-4),log(4)))`.

For each candidate miscoverage `m in {0.05,0.10,0.20,0.30,0.40}`, compute
`q_m` as the conservative finite-sample order statistic of the inner-OOF
normalized residuals `abs(y_cf-mu_cf)/max(sigma,1e-4)`: sorted index
`min(ceil((n+1)*(1-m))-1,n-1)`. Construct `I=[mu-q_m*sigma,mu+q_m*sigma]`.
Decision is BOOST only when L>0, SUPPRESS only when U<0, and KEEP otherwise;
all equality boundaries KEEP. No held-class label, scale, residual, interval,
threshold, or strength may influence this selection.

The operating-point selector evaluates the five candidates only on aggregated
inner-OOF outer-training evidence. A candidate qualifies when coverage >=10%,
accepted opposite-sign rate <=5%, and relative accepted wrong-sign reduction
>=25% versus unfiltered nonzero mu. Among qualifiers select greatest coverage,
then lowest wrong-sign rate, then smaller m. If none qualifies, the fold is
`NO_QUALIFIED_SAFE_OPERATING_POINT` and emits KEEP for its held class.

## Intervention and comparators

The frozen action is an abnormal-logit-only, identical-at-all-stages correction
using the existing R0/Phase2B operator and fixed alpha `0.25`; normal logits
are never changed. Comparators from the same cache are native Phase2B,
direction-only signed alpha 0.25, and interval-selective signed alpha 0.25.
Oracle sign, if serializable from existing R0 evidence, is diagnostic-only and
never selects a controller or operating point.

## Metrics, gates, and branches

Primary outcome is macro pixel-AP delta versus native. Safety metrics are
accepted opposite-sign rate, coverage, and relative reduction versus direction
only. Guardrails are macro pixel-AUROC delta, per-class pixel-AP breadth, and
descriptive worst-class delta. R2 passes only when all audits pass, safety is
<=5% / >=10% / >=25% respectively, macro pixel-AP is strictly greater than
native, at least 9/12 classes have non-regressing pixel-AP, and macro pixel
AUROC delta is no worse than -0.50 percentage points. The AUROC guardrail is
the historical R1/R2 project tolerance, frozen here before R2 evidence.

Any no-qualified point, anti-safety result, downstream failure, numerical or
audit failure is `R2_SCIENTIFIC_STOP` (or `ENGINEERING_STOP` for correctness),
requires terminal evidence, and terminates the main progression. Only an R2
PASS may create R3 under its separate preregistration.

## Firewall and audit contract

Phase2B optimizer steps, added CLIP forwards, MVTec reads, and Medical reads
are exactly zero. The run refuses an existing attempt/summary/terminal marker.
Pre/post audits serialize provenance, P8 protected hashes, cache hashes,
class/image/patch ordering, all fold/inner lists, interval/action reload parity,
decision and metric parity, and firewall counters. Scientific outputs are never
overwritten or rerun.
