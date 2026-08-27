# P32 Optimization Report — Functional Margin Effect

Status: `P32_FORMULATION_AND_IMPLEMENTATION_OPTIMIZED`

This report covers formulation selection, offline optimization, and
engineering qualification only. It does not authorize a scientific Stage 2
run. No scientific optimizer step, CLIP/Phase2B forward, teacher forward,
cache rebuild, or held-result tuning was performed. The 51 optimizer steps
listed below were explicitly engineering smoke/profile steps after the
preregistration freeze.

## 1. Inherited causal mechanism

The P30R1 causal forensic and the P31 native control establish:

```text
primary   = TEACHER_DIRECTION_NOT_CAUSAL
secondary = SPARSE_SELECTIVE_CORRECTION
```

P30 had better raw teacher-direction metrics but much worse detection than
P30R1. P30R1's correction is sparse and anomaly-enriched, but the exact native
control is slightly better on both locked candle endpoints. The problem is
therefore not “make cosine larger.” It is whether any teacher-induced effect
that matters after the frozen deployment operator should be transferred at
all.

## 2. Original candidate formulation

The selected draft formulation was a native-relative deployed margin effect:

```text
Delta_m_s = m(deploy(native + symmetric(student))) - m(deploy(native))
Delta_m_t = m(deploy(native + symmetric(teacher))) - m(deploy(native))
L        = mean SmoothL1(Delta_m_s, stop_gradient(Delta_m_t), beta=1)
```

The exact repository algebra shows that the native logits cancel from the
pre-softmax margin effect. Let `D` denote the existing Gaussian/resize/stage
mean map. The optimized equivalent is:

```text
Delta_m_s = D(mean_stage(student_region))
Delta_m_t = D(teacher_region)
L_FME     = mean SmoothL1(Delta_m_s, stop_gradient(Delta_m_t), beta=1)
```

The final form is mathematically identical to the original effect definition,
but avoids a redundant native-logit forward and avoids applying the sigmoid.
It preserves the exact downstream margin effect rather than matching the
nonlinear probability change, which would be native-margin dependent and could
hide radial changes in saturated regions.

## 3. Simplifications made

1. **Removed native-logit dependence from the objective.** Symmetric two-class
   correction and linear pre-softmax deployment make the native term cancel.
2. **Collapsed three stage transforms to one stage-mean transform.** The
   deployment operator is the same for every stage, so `D(mean_stage(s))` is
   exact. This removes two repeated blur/resize paths from the objective.
3. **Removed all self-normalization.** P30's radial non-identifiability is not
   repeated; effect magnitude remains identifiable.
4. **Kept one inherited robust loss.** `SmoothL1(beta=1.0)` is retained from
   the frozen P30R1 convention. No new epsilon, lambda, beta, threshold, or
   learned scalar is introduced.
5. **Kept the teacher detached.** The teacher is a frozen source cache target.
6. **Kept the native control.** A zero-adapter result remains the safety and
   usefulness comparator; the formulation is not presumed superior to native.

After the preregistration freeze, the production implementation received one
semantics-preserving speed optimization. The fixed `9x9 -> 518x518` transform
is separable, so production precomputes its canonical FP32 one-dimensional
`518x9` matrix once and evaluates the same transform as `A @ x @ A.T`. The
readable reference still executes the existing blur/resize/two-logit
deployment path. This changes no equation, constant, tensor shape, or
objective term; it only removes redundant intermediate work.

## 4. Formulations rejected

| formulation | rejection reason |
|---|---|
| raw residual SmoothL1 | repeats P30R1's teacher-vector target and does not test the forensic mechanism |
| cosine/sign or normalized effect | repairs an internal metric and recreates radial non-identifiability |
| probability-delta loss | depends on native-margin saturation and can make identical probability changes conceal different margin effects |
| L2 effect loss | unbounded tail influence; not needed when inherited SmoothL1 is finite and bounded in derivative |
| global rank/listwise loss | frozen native/P30R1 rank is already highly redundant; pair/list choices add complexity |
| local rank loss | local effect-order agreement is weak and the result is a known ranking family, not an isolated causal target |
| learned gate or confidence head | source-only support/magnitude cannot identify held usefulness; adds a module and parameters |
| fixed L1 sparsity penalty | sparsity is descriptive evidence, not proof that L1 is the causal mechanism |
| teacher-at-inference or auxiliary decoder | violates zero incremental inference-cost target and is unnecessary for the declared effect |

## 5. Synthetic adversarial results

The deterministic suite used the exact fixed operator and FP32 tensors. It
included exact zero, zero-teacher/nonzero-student, near-zero teacher, normal
scale, 0.01×/0.1×/1×/10×/100× scale mismatch, sign reversal, 1% sparse
corruption, one extreme heavy-tail sample, mixed scales from 0 to 100×,
all-null behavior, and a high-confidence uniform intervention.

Observed properties:

- every case had finite loss and finite gradients;
- exact zero had loss and gradient zero;
- zero teacher with nonzero student had a finite restoring gradient;
- all non-equal scale/sign cases had positive descent alignment toward the
  teacher effect;
- 10× and 100× cases remained finite with bounded SmoothL1 derivative;
- mixed-scale max/median per-sample gradient ratio was `20.6361` in the
  deterministic stress batch, with no NaN/Inf;
- the one-outlier and 1% sparse cases had a large relative per-sample
  gradient contrast only because the other samples were exact zero-loss
  cases; the outlier gradient itself remained finite and bounded;
- the analytic per-coordinate bound at batch size one is
  `max_column_sum(D)/(518*518*3) = 0.00531713`.

The full machine-readable cases, gradients, nonzero fractions, descent
alignment, and dominance ratios are in
`research/sabra_v2/region_distill/P32_PREFLIGHT_FALSIFICATION.json`.

## 6. Frozen source-only robustness

The unique source Tier-B cache union contained 2,162 samples from 23,782
exposures; duplicate values agreed exactly. The fixed deployment operator has
rank 81 and condition number `6.14036`, so it is numerically well-conditioned
on the shared 9×9 target. Source teacher/effect RMS values were:

| quantity | q01 | q50 | q99 |
|---|---:|---:|---:|
| raw teacher RMS | 0.196889 | 2.056865 | 4.960112 |
| deployed margin-effect RMS | 0.160942 | 1.989157 | 4.960111 |

The exact-zero sample fraction was `0.004163`; the fixed 256-pixel probe
effect mean absolute value averaged `2.570551`. Source category medians vary,
but no category-specific threshold is used or proposed. The inherited
P30R1 scale denominator retains its analytic inverse-weight q99/q01 spread of
`24.5616`; P32 does not carry that denominator into the objective.

Source-only evidence establishes support and effect scale. It cannot establish
held actionability, so no source-derived gate is included.

## 7. Identifiability and expected complexity

The effect objective identifies the deployed effect and, for a shared teacher
target, the stage-mean 9×9 residual because the fixed operator is full-rank.
It intentionally does not constrain stage-specific components orthogonal to
the stage mean. This is a declared downstream invariance, not an accidental
loss of information.

The final objective is O(BHW) with vectorized tensor operations, no pairwise
O(N²) computation, no Python per-sample/pixel loop, no auxiliary network, no
teacher forward, and no inference-time computation beyond the existing
adapter/deployment path. One stage-mean transform replaces three redundant
effect paths, and the production transform is evaluated by the fixed
separable matrix. Expected incremental inference overhead is `0%`; cached
training overhead is measured separately from DataLoader/cache-read time.

## 8. Hyperparameter accounting

| scalar | classification | value |
|---|---|---:|
| SmoothL1 beta | inherited/frozen | 1.0 |
| image size | inherited/frozen | 518 |
| blur kernel/sigma | inherited/frozen | 7 / 1.0 |
| resize alignment | inherited/frozen | `align_corners=True` |
| new lambda | genuinely new | none |
| new epsilon | genuinely new | none |
| new threshold | genuinely new | none |
| category-specific scalar | genuinely new | none |

`new tuned hyperparameters = 0`.

## 9. Overconstraint and null-control assessment

The overconstraint risk is `MEDIUM-LOW`: the effect transform is a new
training target, but it is one objective with no auxiliary constraints. The
native/zero-adapter control has `LOW` risk and remains mandatory. If native is
non-inferior in the future locked comparison, P32 must conclude that teacher
imitation is unnecessary for the scope rather than adding another method.

## 10. Freeze and engineering decision

The authoritative equation is frozen in `P32_PREREGISTRATION.md` with
SHA-256
`5141722b2c3e3d3aac721390a8943d54356dd17bdfdad8aaa6bd7302766a5cc2`.
Production/reference parity and the cached engineering path passed. The
measured warmed comparable cached step was `0.004818620283156634` seconds
versus the frozen P30R1 reference `0.004393984079360962` seconds, a `9.664%`
overhead. The objective-only median was `0.0002380959987640381` seconds versus
`0.00022779200226068496`, a `4.523%` overhead. Full DataLoader/cache-read
latency is reported separately because the frozen P30R1 timing excludes
DataLoader wait. The 40-step profile was finite and the native/control
scientific gates were not run.

Engineering qualification used 51 optimizer steps solely for smoke/profile,
not science. The current scientific counts remain:

```text
new scientific Stage 2 attempts = 0
new Stage 3 attempts             = 0
full runs                        = 0
held-result tuning iterations   = 0
new CLIP forwards                = 0
new Phase2B forwards             = 0
cache rebuilds                   = 0
scientific optimizer steps       = 0
engineering optimizer steps     = 51
```

`P32_OPTIMIZATION_AND_ENGINEERING_COMPLETE`
