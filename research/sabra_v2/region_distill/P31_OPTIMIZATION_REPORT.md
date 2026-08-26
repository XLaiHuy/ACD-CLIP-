# P31 Optimization Report — Native / Zero-Adapter Control

Status: `OPTIMIZATION_COMPLETE`

This report covers formulation optimization and non-held engineering design
only. It does not authorize a scientific held run.

## 1. Inherited mechanism and selected hypothesis

The frozen P30R1 causal forensic completed with:

```text
PRIMARY_MECHANISM   = TEACHER_DIRECTION_NOT_CAUSAL
SECONDARY_MECHANISM = SPARSE_SELECTIVE_CORRECTION
```

The selected hypothesis was:

```text
P31_NATIVE_ZERO_ADAPTER_CONTROL
```

The forensic showed that P30's higher direction fidelity did not produce
better detection, P30R1 stayed close to native while direction collapsed, and
the exact native counterfactual was slightly better than P30R1 on both reported
pAP and pAUROC. The selected causal question is therefore whether any teacher
intervention is needed before designing another learned transfer mechanism.

## 2. Mechanism sanity check

1. **Direct mechanism test:** yes. Zero intervention directly tests whether the
   teacher residual contributes downstream utility at all; it does not replace
   the failed direction target with another metric.
2. **Causal versus metric repair:** causal. The comparison is native output
   versus an existing teacher-residual intervention, not cosine/sign repair.
3. **Unnecessary teacher imitation:** removed. P31 has no teacher target,
   teacher forward, teacher cache dependency, or teacher-at-inference path.
4. **Old SABRA pattern:** avoided. There is no sum of objectives, auxiliary
   constraint, scale, gate, ranking term, or category-specific parameter.
5. **Simpler equivalent:** none. `r=0` is the minimal exact control for the
   do-not-intervene question.
6. **Native baseline strength:** required. The frozen native result is already
   comparable or better, so a learned method would be premature.

## 3. Original draft formulation and simplification

The previous draft proposed an evaluation-only native control:

```text
r_P31(x) = 0
ell_P31(x) = ell_native(x)
A_P31(x) = A_native(x)
```

The formulation was not expanded. The optimization pass removed every
operation that could be mistaken for a scientific method:

- no adapter construction or forward;
- no residual normalization;
- no detach or gradient path because there is no objective;
- no teacher residual, score delta, gate, threshold, or confidence rule;
- no duplicate full-resolution map;
- no pairwise or per-pixel Python computation;
- no learned parameter and no optimizer state.

The final implementation is a pure immutable-output control. Given a finite
native output array `N`, the control returns an exact copy `C = copy(N)` and
the diagnostic delta `C - N = 0`. A zero residual is represented only in the
synthetic contract; it is not sent through a model.

## 4. Rejected formulations

### Direct teacher residual imitation

Rejected because it repeats the forensic primary failure: P30's strong
direction metric was not downstream-causal, and P30R1's useful detection
behavior did not require direction. It also creates an unnecessary teacher
dependency.

### Native-relative downstream-logit-effect transfer

Retained only as a future research alternative. It is a plausible one-loss
candidate, but the current native control has already matched or exceeded the
existing adapter without training. A teacher-effect target is not justified
until the null control fails. No target was constructed and no held result was
used to tune it.

### Rank/margin transfer

Rejected for this phase. Rank transfer has prior art, requires pair/tie
conventions, risks an O(N²) path, and was not isolated by the forensic. Adding
it now would optimize a metric rather than test the identified causal claim.

### Learned gate, sparsity penalty, or identity adapter

Rejected as unnecessary. The zero control already gives exact abstention and
the lowest-cost falsification of useful intervention versus damage avoidance.

## 5. Synthetic adversarial suite

The deterministic suite contains 15 cases: exact zero, near-zero, normal
scale, `0.01x`, `0.1x`, `1x`, `10x`, `100x`, sign flip, 1% sparse corruption,
heavy-tail corruption, mixed-scale batch, one extreme outlier sample, all-null
no-intervention, and high-confidence intervention.

For every case the control output is exactly the native input, so the loss is
not defined (`objective_count=0`), output delta is exactly zero, student
gradient is exactly zero, expected update is `NO_UPDATE`, and no sample can
dominate a batch. All outputs are finite, including the finite synthetic
heavy-tail and outlier values. The complete per-case record is in
`P31_PREFLIGHT_FALSIFICATION.json`.

This is the expected behavior, not a learned stability claim: an identity
control cannot be destabilized by an attempted residual because it never
consumes that residual.

## 6. Source-only robustness

The existing `/workspace/p27r1_cache_v1` source cache was inspected without
held labels, held masks, cache rebuild, or model forward. All 12 Tier-A native
logit arrays and all 12 Tier-B teacher-region arrays were finite. The audit
found:

- Tier-A native logits: `17,758,668` values, absolute q99 approximately
  `9.88314`, maximum approximately `9.90699`;
- Tier-B teacher regions: `1,926,342` values, absolute q99 and maximum
  approximately `4.96011`, exact-zero fraction approximately `0.42255`;
- source labels read: `0`;
- source masks read: `0`;
- new model forwards: `0`.

These distributions do not tune a P31 constant. They demonstrate that the
control remains finite and independent of source scale, zeros, tails, and
category dispersion.

## 7. Hyperparameter minimization

| Constant | Classification | Final treatment |
|---|---|---|
| Residual value | analytic | exactly `0` |
| Objective count | analytic | exactly `0` |
| Non-inferiority margin | analytic protocol rule | exactly `0.0` metric units |
| Native reconstruction tolerance | inherited engineering tolerance | `2e-5`, not a scientific tuning parameter |
| Any teacher scale/epsilon/temperature | not required | removed |
| Any gate/ranking threshold | not required | removed |

New tuned hyperparameters: `0`.

## 8. Final frozen formulation

For every cached native tensor or output `N`:

```text
R_P31 = 0                       (conceptual residual; no model call)
C_P31(N) = copy(N)              (exact native control output)
Delta_P31(N) = C_P31(N) - N = 0
L_P31 = undefined               (zero objectives)
grad_P31 = 0                    (zero optimizer path)
```

At the stored deployment shapes, native logits are `[3, B, 1369, 2]` and
native anomaly maps are `[B, 518, 518]`. The control supports arbitrary finite
leading dimensions and preserves dtype; no shape-specific architecture is
introduced.

## 9. Complexity and overconstraint

The hot operation is a linear O(N) immutable copy, or a zero-copy read where
the caller already owns the native output. There is no pairwise operation, no
Python per-pixel loop, no neural forward, no teacher path, and no inference
branch. Expected new inference overhead is `0%`; training overhead is `0%`
because there is no training.

Overconstraint risk is `LOW`: P31 has one null mechanism, zero objectives, zero
new parameters, zero tuned scalars, and no teacher imitation. It does not
repeat P29's mixed-objective conflict, P30's direction-only identifiability
problem, or P30R1's scale-reweighted residual objective.

## 10. Decision

The formulation is frozen for implementation as an evaluation-only native
control. The future scientific comparison remains exactly one locked native
versus P30R1 outcome comparison. A control failure would require a new
research decision; it would not authorize a downstream-effect loss or ranking
loss automatically.

`PASS_TO_PREREGISTRATION_FREEZE`
