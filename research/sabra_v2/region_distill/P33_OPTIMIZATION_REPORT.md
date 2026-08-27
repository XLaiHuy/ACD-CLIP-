# P33 Optimization and Preflight Report

Status: `P33_FORMULATION_OPTIMIZED`

This report covers frozen/offline formulation work and engineering-only
checks. It does not authorize or contain a P33 scientific Stage 2 result.

## 1. Mechanism inherited from the forensic

P32 was radially safe but meaningfully dense in residual space: `87.1481%`
of coordinates exceeded the inherited correction threshold versus `11.1358%`
for P30R1, with effective support `52.5447%` versus `5.6409%`. P32 retained
most P30R1 support and added broad low-level support. Its score-effect floor
was also denser (`3.32%` versus `77.53%` at the `1e-10` near-zero criterion).
The frozen rank analysis showed broader displacement but did not isolate a
smallest-gap-specific flip excess. H1 lost selectivity, with H4 support
expansion and H5 actionability-not-magnitude secondary, is therefore the
selected causal mechanism.

## 2. Selected hypothesis and original formulation

`CONTINUOUS_ACTIONABILITY_WEIGHTED_FUNCTIONAL_TRANSFER` tests whether the
functional margin effect should be learned only where the frozen teacher
requests an operational effect. The original research-decision formulation
was:

```text
E_s = D(mean_stage(student_region))
E_t = D(teacher_region)
w   = clip(abs(E_t) / C, 0, 1)
L   = mean(w * SmoothL1(E_s, stop_gradient(E_t), beta=1, reduction=none))
```

with inherited `C=4.960109710693359`.

## 3. Simplifications and rejected formulations

The final formulation removes or refuses all additional mechanisms:

- no learned gate, classifier, threshold, coverage term, or inference branch;
- no target replacement by `w*E_t`, so signed magnitude remains identifiable;
- no L1, sign, ranking, normal-region, calibration, or auxiliary loss;
- no category-specific parameter or new tuned scalar;
- one fixed deployment transform and one weighted SmoothL1 objective.

Candidate B, `SmoothL1(E_s,w*E_t)`, was rejected because it shrinks the
functional target and creates a restoring null-target pressure when all
teacher effects abstain. Candidate C, hard support transfer using the
inherited raw threshold, was rejected because its intervention is
discontinuous and threshold-sensitive. Both alternatives passed basic finite
arithmetic but failed the mechanism/simplicity filter.

## 4. Synthetic adversarial result

The deterministic suite in
[`P33_PREFLIGHT_FALSIFICATION.json`](P33_PREFLIGHT_FALSIFICATION.json)
covered exact/near zero, `0.01×/0.1×/1×/10×/100×`, sign reversal, 1% sparse
support, heavy tails, mixed scales, one extreme outlier, all abstain, and
high-confidence active effects. All selected-objective cases were finite;
weights were bounded and detached, and the largest observed gradient
component was `0.0051648314`. Exact zero and all-abstain had exactly zero loss
and gradient. The mixed-scale maximum/nonzero-median per-sample gradient
ratio was `12.7847`; the one-outlier ratio was `2.2493`.

The selected formulation therefore has an explicit abstention interpretation:
zero teacher effect contributes no correction-learning signal. It does not
pretend that source effect magnitude is a validated anomaly-utility oracle;
that remains the single future scientific question.

## 5. Source-only robustness

The immutable Tier-B source union contains `2,162` unique samples and
`23,782` exposures. The actionability weight computed from the frozen
deployed teacher effect had mean `0.51861`, exact-zero fraction `0.21364`,
and range `[0,1]`. No held labels, masks, metrics, new cache, or neural
forward were used. Category-specific rules were not used. Cross-stage
consistency and calibrated teacher confidence were unavailable and were not
invented.

The source audit establishes only an operational teacher-requested-effect
descriptor. It does not establish future anomaly utility, so the native
control and locked endpoint gates remain mandatory.

## 6. Final frozen equation and conditioning

The authoritative equation is in `P33_PREREGISTRATION.md`, whose SHA-256 is
`d2460555be14af7d23316e43ad16c8585faeecbedf1698ee71f29dce765aed6c`.
Production uses the algebraically equivalent fixed separable `518x9` map
already used by P32; the additional P33 work is bounded elementwise weighting.
All teacher-derived tensors are detached, the target is signed and
unshrunk, and the student is not normalized.

## 7. Runtime design

The objective remains O(`B*518*518`) after the existing two matrix effects,
with O(`B*518*518`) elementwise absolute-value, clamp, SmoothL1, and weighted
mean work. There is no pairwise operation, Python per-pixel loop, additional
network, teacher forward, inference-time weight, or cache rebuild. The
engineering profile reports objective cost separately because the new
weighting is scientifically required; input/cache time dominates the
end-to-end path.

## 8. Overconstraint and P32 lesson check

Risk is `LOW`: one mechanism, one objective, zero new tuned scalars, no
category-specific behavior, and zero incremental inference cost. P30's scale
lesson is preserved by retaining the signed target and avoiding self-
normalization. P30R1's direction lesson is preserved by not matching raw
direction globally. P31's native safety lesson is preserved by a mandatory
native control. P32's lesson is addressed directly by conditioning where
functional transfer is active.

`P33_FORMULATION_OPTIMIZED`
