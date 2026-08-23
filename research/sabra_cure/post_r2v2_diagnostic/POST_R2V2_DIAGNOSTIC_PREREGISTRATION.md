# SABRA-CURE POST-R2V2 ACTIONABILITY DIAGNOSTIC V1

Status: `FROZEN_BEFORE_DIAGNOSTIC_EXECUTION`.

This is one deterministic, post-hoc diagnostic of the published R2-v2 terminal
study, parent `f097be019de365a9598551b4c3c97e33e3d39583`.  It is not an R2-v2
rerun, candidate, threshold/alpha study, or scientific benchmark.  R2-v2
remains `R2V2_SCIENTIFIC_STOP`; its fixed actuator remains alpha `.25`.

## Inputs and firewall

Only the immutable VisA source-side R2-v2 fold outputs, parameters, native
cache evidence, masks, and already-authorized R0/R1/R2 artifacts may be read.
No MVTec or Medical read, CLIP forward, Phase2B optimization, representation
training, threshold search, coverage tuning, alpha comparison, or R2-v3/R3/R4
execution is permitted.  The terminal R2-v2 outputs are hash-protected and
never written.

## Fixed diagnostic reconstruction

For every one of the 12 persisted held classes, `proposal=sign(mu)` and the
published harm-aware action is the persisted `actions`.  Correct/wrong uses
`sign(action)==sign(utility)` for nonzero action; zero-utility actions are
reported separately and excluded from both labels.  Fixed-alpha deployment is
the frozen abnormal-logit correction exactly as R2-v2 used it.

The only target-aware counterfactual action cohorts are labelled
`POST_HOC_ORACLE_DIAGNOSTIC`:

- D0: native (KEEP everywhere);
- D1: persisted R2-v2 harm-aware actions;
- D2: accepted sign-correct actions only;
- D3: accepted sign-wrong actions only;
- D4: rejected sign-correct proposals only.

These cohorts explain the historical result only; none is deployable and none
changes the R2-v2 decision.  Metrics are computed at class/image/cohort level;
there is explicitly no per-patch AP attribution.

## Frozen hypotheses and measurements

H1 tests whether D2 is beneficial while D3 erases the D1 gain.  H2 describes
D2 relative to native as beneficial, weak/neutral, or harmful.  H3 reports
Pearson and Spearman (descriptive, n=12) between per-class harm reduction,
wrong-sign reduction, coverage, pAP delta, pAUROC delta, and loss delta.  H4
uses D4 solely to quantify oracle-rejected correct-direction reservoir.

H5 partitions already-persisted GT-free variables (`native_score`, within-image
native score rank, `abs(mu)`, harm risk, sigma, signed native margin, and
action type) using exactly five pooled equal-frequency bins for continuous
variables.  Ties use deterministic `searchsorted(..., side='right')`; repeated
edges create empty bins rather than a changed bin count.  H6 uses action count,
action fraction, 4-neighbour density, and high-score concentration per image.
H7 compares class-level frozen focal+Dice loss delta, pAP delta, and pAUROC
delta.  All correlations are descriptive only.

## Predeclared decision rules

`CALIBRATION` is not studied here.  H1 is SUPPORTED when D2 pAP exceeds native
and D3 is below native, with D1 non-positive versus native; otherwise it is
MODERATE/WEAK by the same directional evidence.  H3 is SUPPORTED only when
both rank/linear harm-vs-pAP association are weak in magnitude (<.30) or
opposite in direction; n=12 prevents causal claims.  H4 is SUPPORTED when D4
has positive class-macro pAP delta and at least six classes have positive D4
delta.  H5/H6/H7 require concordant descriptive evidence in at least six
classes; otherwise they are PLAUSIBLE/WEAK/INSUFFICIENT_EVIDENCE.

Target families are assessed, never trained: T0 `abs(y)` reference; T1 local
loss benefit; T2 a bounded image/class ranking-surrogate delta (pAUROC and
positive-vs-negative ordering change under a fixed action); T3 image/cohort
action value.  `GO_RANKING_ALIGNED` requires a written target definition,
nested-LOCO source construction, GT-free inference path, better empirical
alignment with pAP than T0/T1, and support in at least six classes.  Failure of
any requirement yields `IMAGE_LEVEL_ONLY`, `NO_GO`, or `UNRESOLVED`, never an
automatic new method.

Exactly one execution is marked by `ATTEMPT_STARTED.json`; an engineering
failure after that marker is terminal for this diagnostic.
