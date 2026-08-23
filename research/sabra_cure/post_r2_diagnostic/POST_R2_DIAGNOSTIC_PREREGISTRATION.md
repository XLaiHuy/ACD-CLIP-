# SABRA-CURE Post-R2 Diagnostic Preregistration v1

Status: `FROZEN_BEFORE_DIAGNOSTIC_EXECUTION`

Base terminal R2 commit: `7785fb2d984a226f14eccfc387bd537ff7d7957b`.
This is one deterministic, source-only, post-hoc diagnostic execution. It is
not an R2 rerun, a new scientific model, or evidence that changes
`R2_SCIENTIFIC_STOP`.

## Inputs and immutability

Inputs are the published R2 summary, post-audit, fold/parameter/inner-crossfit
evidence; immutable R0 utility and alpha evidence; immutable R1 fold evidence;
and the existing source/Trust caches. R2 actions, `y`, `utility`, `mu`, and
`sigma` are read from the R2 folds and must reconstruct from their published
parameters. Native cached logits/probabilities and authorized VisA masks may
be reopened solely to reconstruct the already-persisted R2 action deployment
and exact ranking descriptions. No CLIP forward, Phase2B update, MVTec read,
or Medical read is permitted.

The audit records SHA256 hashes of every R2 fold, parameter, inner-crossfit,
summary, post-audit, R0 alpha-selection, R0 utility, and R1 fold input. It
fails if any protected R0/R1/R2 path differs from the base commit.

## Frozen hypotheses

- **H1 Wrong-sign target too narrow.** Compare all persisted actions with the
  post-hoc oracle subset containing only accepted actions whose sign matches
  frozen R0 utility. Material harm from that subset means wrong-sign safety is
  insufficient.
- **H2 BOOST/SUPPRESS asymmetry.** Compare the post-hoc oracle BOOST-only and
  SUPPRESS-only accepted-action cohorts and their exact ranking summaries.
- **H3 Actionability/utility-strength mismatch.** Contrast accepted and KEEP
  patches by raw absolute utility, `abs(mu)`, `sigma`, interval width
  `2*q*sigma`, frozen relational features, native patch score, action density,
  and sign correctness.
- **H4 Target-loss/pAP ranking mismatch.** Test whether the sign-correct oracle
  cohort improves the frozen R0 loss while reducing exact pAP; this is the
  direct, post-hoc criterion for a target/ranking mismatch.
- **H5 Fixed-alpha/selected-subset interaction.** Describe only the already
  persisted global R0 signed-alpha rows. No alternative-alpha deployment is
  computed for any R2 subset.
- **H6 Spatial/ranking coupling.** Describe spatial action concentration,
  native patch-score ranks, stage disagreement, peer agreement, and exact
  positive/negative ranking transitions under the persisted action deployment.

## Derived variables and partitions

All patch aggregates use every persisted R2 outer-held patch, in committed
class/image/patch order. Accepted means `actions != 0`; KEEP means zero.
For accepted patches, sign-correct is `actions * sign(utility) > 0`, sign-wrong
is `< 0`, and utility-near-zero is `abs(utility) <= 1e-8` (reported separately,
never silently classed as correct). BOOST and SUPPRESS are action values `+1`
and `-1`.

Five pooled descriptive equal-frequency bins are independently derived from
the full persisted patch population for `abs(utility)`, `abs(mu)`, `sigma`,
interval width, native patch score, stage-disagreement feature, and peer-
consensus feature. NumPy linear quantiles at `[0,.2,.4,.6,.8,1]` are frozen;
values equal a boundary use the higher bin except the maximum. The emitted
boundaries are descriptive, not selected after outcomes. Raw `abs(y)` is
reported with fixed bins `[0, .1, .25, .5, .75, 1]` as a secondary target-scale
description.

Spatial concentration is the fraction of horizontal/right and vertical/down
37x37 patch-grid adjacent pairs for which both patches are accepted, and the
per-image action count. Native patch score is the mean of each exact 14x14
block of the persisted 518x518 native abnormal probability. Per-image AP is
computed only for images containing both labels; unavailable images are null.

## Frozen oracle deployment/reconstruction matrix

The following are explicitly `POST_HOC_ORACLE_DIAGNOSTIC` whenever a cohort is
selected using utility labels. They are counterfactual descriptions only and
are never candidates or gates:

1. `D0_NATIVE`: zero correction.
2. `D1_PERSISTED_R2`: all persisted R2 actions, reconstructed exactly.
3. `D2_SIGN_CORRECT_ONLY`: only accepted sign-correct actions.
4. `D3_SIGN_WRONG_ONLY`: only accepted sign-wrong actions.
5. `D4_BOOST_ONLY`: only persisted BOOST actions.
6. `D5_SUPPRESS_ONLY`: only persisted SUPPRESS actions.

Each condition reports exact class/macro pAP, pAUROC, mean frozen loss,
positive-vs-negative ordering probability, rank displacement by label, and
per-image AP where defined. AP is global and non-additive: no per-patch AP
attribution is calculated. If `D1` pAP delta is negative, the descriptive
retained-harm ratio is `D2 pAP delta / D1 pAP delta`; it is explicitly not an
additive causal decomposition.

## Classification rules

`SUPPORTED` means the frozen direct descriptive criterion below is met;
otherwise use `WEAK`, `PLAUSIBLE`, `INSUFFICIENT_EVIDENCE`, or `NOT_SUPPORTED`
as appropriate without changing a criterion after execution.

- H1 is `SUPPORTED` if D2 pAP delta is negative and its retained-harm ratio is
  at least .50. H1 is `WEAK` if D2 is non-negative.
- H2 is `SUPPORTED` if exactly one of D4/D5 has negative macro pAP delta and
  the opposite condition is non-negative; otherwise `PLAUSIBLE` when their
  pAP deltas differ by at least .005, else `WEAK`.
- H3 is `SUPPORTED` if accepted patches have lower median absolute utility than
  KEEP patches and the accepted bottom two utility bins contain at least 50% of
  actions; otherwise `PLAUSIBLE` if only one holds, else `WEAK`.
- H4 is `SUPPORTED` if D2 mean loss improves versus D0 while D2 macro pAP
  declines versus D0; otherwise `PLAUSIBLE` if the all-action D1 condition has
  that pattern, else `INSUFFICIENT_EVIDENCE`.
- H5 is `INSUFFICIENT_EVIDENCE` unless existing R0 rows directly establish a
  fixed-alpha conflict for the selective subset; no new alpha result is made.
- H6 is `PLAUSIBLE` if accepted-action adjacency exceeds the pooled action-rate
  independence expectation by 25% or more and ranking degradation is present;
  otherwise `WEAK`.

The primary root cause is `MIXED_FAILURE` if two or more hypotheses are
supported/plausible and neither is uniquely direct; otherwise choose the
single matching label from the authorized root-cause menu. If H4 is supported,
the recommended future direction is `ACTION-HARM / RANKING-AWARE SELECTIVE
INTERVENTION`, for user review only.

## Audit, stop, and firewall

Pre/post audits require exact base ancestry, one R2 attempt, R2 post-audit
PASS, parameter/action reconstruction parity, class/image/patch alignment,
finite inputs, aggregate recomputation parity, protected-history immutability,
and zero firewall counters. The script writes only under
`results/sabra_cure/post_r2_diagnostic/` and
`research/sabra_cure/post_r2_diagnostic/`.

This diagnostic stops after its terminal evidence is published. It does not
create R3, R4, MVTec, Medical, an alpha sweep, a learned controller, or a new
scientific preregistration.
