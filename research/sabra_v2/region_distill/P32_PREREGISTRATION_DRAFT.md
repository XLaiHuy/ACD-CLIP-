# P32 Preregistration Draft — Functional Margin Effect

Status: `P32_PREREGISTRATION_DRAFT_ONLY`

This document is a draft for one future scientific Stage 2 comparison. It is
not execution authorization. It creates no scientific UUID, execution marker,
or final experiment hash. No P32 scientific run has started.

## 1. Scientific hypothesis

`SELECTED_P32_HYPOTHESIS = FUNCTIONAL_MARGIN_EFFECT`

The primary causal claim is:

> When raw teacher correction direction is not the causal downstream quantity,
> a student trained to match the teacher's exact native-relative deployed
> margin effect can transfer useful intervention without being constrained to
> imitate the teacher residual vector in raw 243-coordinate space.

The native/zero-adapter detector is a mandatory control. The hypothesis does
not assume that teacher intervention is beneficial; if native is non-inferior,
the scientific conclusion is that the learned intervention is unnecessary for
the locked scope.

## 2. Mechanism and exact formulation

For a scalar 9×9 residual `r[g]` at stage `g`, define the frozen deployment
operator

```text
D(r) = mean_g Interpolate_37_to_518(
         GaussianBlur_7,sigma=1(Interpolate_9_to_37(r[g])))
```

`Interpolate_9_to_37` and `Interpolate_37_to_518` are bilinear with
`align_corners=True`. The Gaussian kernel and sigma are the existing
Industrial deployment values. The two-class correction is symmetric:

```text
normal_logit  += -r / 2
abnormal_logit +=  r / 2
```

Therefore, for student residual `s ∈ R^[3,B,9,9]` and teacher target
`t ∈ R^[B,9,9]`,

```text
Delta_m_student = D(s) = D(mean_stage(s))
Delta_m_teacher = D(t)
m_student       = m_native + Delta_m_student
p_student       = sigmoid(m_student)
```

The only scientific objective is

```text
L_FME = mean SmoothL1(
    Delta_m_student,
    stop_gradient(Delta_m_teacher),
    beta=1.0,
    reduction="mean"
)
```

The teacher is detached. The student is not self-normalized. There is no
cosine, sign, Pearson, Spearman, ranking, sparsity, normal-pixel, calibration,
segmentation, feature, gate, or auxiliary term. `beta=1.0` is inherited from
the frozen P30R1 robust-regression convention, not tuned for P32.

The fixed operator has shape `[268324,81]`, rank `81`, and condition number
`6.14036` on the frozen geometry. P32 therefore tests deployment-aware
conditioning of the stage-mean residual; it does not claim to recover a
mathematical nullspace or make radial scale unidentifiable.

## 3. What remains frozen

- canonical Phase2B parent, architecture, prompt, image size `518`, three
  stages, and symmetric margin construction;
- unchanged `RegionResidualAdapter`, including zero-output initialization;
- frozen Tier-A/Tier-B cache root and provenance;
- teacher target semantics and source-only teacher cache;
- one locked LOCO candle fold: 1,962 fit records and 200 held records;
- 20 epochs, batch size 1, seed 0, canonical FP32 and deterministic policy;
- AdamW learning rate `0.001`, betas `(0.9,0.999)`, epsilon `1e-8`, weight
  decay `0.01`, and no AMSGrad;
- no new scalar hyperparameters, no category-specific parameters, and no
  schedule or optimizer change;
- native/zero-adapter control and frozen P30R1 output comparator;
- fixed score postprocessing and pAP/pAUROC implementation;
- one scientific attempt only; no automatic rerun or fallback method.

## 4. Allowed and forbidden data

Allowed during future training:

- VisA fit/source samples in the locked candle fold;
- frozen Tier-A segmentation features and native logits;
- frozen Tier-B teacher region targets;
- immutable metadata needed for source identity and cache validation.

Forbidden before prediction freeze:

- held GT labels or held masks;
- held outcome metrics or any held-derived threshold/weight;
- any new source/held cache rebuild;
- any teacher, CLIP, or Phase2B recomputation;
- any class-specific actionability rule.

Held labels and masks may be read only after the candidate and native control
prediction artifacts are frozen, solely to calculate the predeclared final
outcomes and descriptive post-freeze diagnostics.

## 5. Allowed model forwards and deployment cost

The future cached training path may execute one adapter/student forward per
fit batch. It may reuse cached Tier-A features and native logits. It may not
execute a new CLIP or Phase2B forward, and it may not execute a teacher neural
forward. The functional objective is training-only.

At inference, the path is the existing adapter plus the existing deployment
operator. There is no teacher, extra model, gate, ranking branch, or iterative
refinement. Incremental inference overhead relative to the existing adapter
path is preregistered as `0%`; relative to native, the already-existing adapter
cost is not attributed to P32.

## 6. Objective and hyperparameter contract

| item | locked value |
|---|---:|
| scientific objective count | 1 |
| new tuned scalar hyperparameters | 0 |
| SmoothL1 beta | inherited `1.0` |
| new learnable parameters | 0 |
| teacher detach | yes |
| student self-normalization | no |
| category-specific parameters | 0 |
| teacher inference | forbidden |
| inference overhead | 0% incremental |

## 7. Preflight and engineering gates

All gates must pass before a scientific Stage 2 marker can ever be created:

1. the frozen equation and repository deployment operator agree exactly;
2. the fixed operator rank/conditioning and tensor shapes are unchanged;
3. all deterministic synthetic adversarial cases are finite;
4. exact zero has zero loss/gradient, zero-teacher nonzero-student has a
   finite restoring gradient, and near-zero targets do not produce NaN/Inf;
5. 0.01×, 0.1×, 1×, 10×, and 100× scale cases have the expected radial
   ordering and update direction;
6. sign reversal, 1% sparse corruption, heavy-tail corruption, mixed-scale
   batches, one extreme outlier, all-null, and high-effect cases have finite
   loss/gradients and bounded SmoothL1 influence;
7. source-only scale/support/tail statistics pass without category-specific
   tuning;
8. production/reference objective output and student gradient agree within
   the explicitly recorded FP32 tolerance;
9. the cached production smoke confirms teacher/frozen tensors are unchanged,
   the student update is nonzero, no held data is read, and no cache or model
   recomputation occurs;
10. training median/end-to-end overhead is ≤10% preferred and ≤15% maximum
    versus the closest cached P30R1 path, with objective and input/cache costs
    reported separately; unexplained >15% is an engineering stop;
11. peak memory growth is ≤10% unless a documented fixed tensor explains it;
12. the native control and candidate use identical locked data identity and
    scoring/postprocessing.

Failure of a preflight or engineering gate stops P32 and does not authorize a
scientific run.

## 8. Future scientific Stage 2 criteria

The outcome comparison is locked before any future P32 held result. The
primary criterion is candidate pAP and the secondary criterion is candidate
pAUROC. The zero-margin native non-inferiority thresholds are the frozen P31
native candle references:

```text
pAP    >= 0.5141403049313743
pAUROC >= 0.9806671435137679
```

The candidate must also satisfy the inherited safety/health criteria:

```text
global residual absolute q99       <= 8.643353872299194
normal score/effective-margin q99  <= 0.0010011587851122385
nonfinite loss count               = 0
nonfinite gradient count           = 0
```

The frozen P30R1 comparator is reported, but raw residual cosine, sign,
Pearson, Spearman, and candidate-vs-teacher ranking are diagnostics only and
are not optimization or pass gates. The functional effect is the declared
mechanism; it is not replaced by a metric-rescue criterion.

If either native non-inferiority endpoint or any health/safety/provenance gate
fails, the hypothesis is falsified for the locked candle scope and P32 stops.
If all gates pass, the result supports retaining the functional-effect
hypothesis for the declared scope, while still making no cross-category claim.

## 9. Scientific stop and no-rerun rules

Stop without interpretation for any data-access, provenance, forbidden-read,
model-forward, cache, shape, numerical, checkpoint, or runtime violation.

There is exactly one future scientific Stage 2 attempt. A failed or
ambiguous result does not allow changed beta, normalization, threshold,
optimizer, schedule, seed, class, sample subset, loss, ranking rule, gate, or
architecture. A repeat is allowed only to repair a documented integrity
failure with identical frozen inputs and code; such a repair is not a new
scientific attempt. Any scientific semantic change requires a new protocol
and preregistration.

## 10. Authorization boundary

This draft authorizes no training and no scoring. It contains no scientific
UUID, execution marker, or final experiment hash.

```text
new scientific Stage 2 attempts = 0
new Stage 3 attempts             = 0
full runs                        = 0
held-result tuning iterations   = 0
new CLIP forwards                = 0
new Phase2B forwards             = 0
cache rebuilds                   = 0
optimizer steps                  = 0
```

`P32_PREREGISTRATION_DRAFT_ONLY`
