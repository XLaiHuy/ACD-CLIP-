# P33 Preregistration — Continuous Selective Actionability

Status: `P33_PREREGISTRATION_FROZEN`

This is the authoritative preregistration for one possible future P33
scientific Stage 2 comparison. It is not an execution marker and does not
authorize a scientific run. The companion JSON records the byte-level SHA-256
of this Markdown file. No scientific UUID or execution marker is created at
freeze.

## 1. Scientific hypothesis

`SELECTED_P33_HYPOTHESIS = CONTINUOUS_ACTIONABILITY_WEIGHTED_FUNCTIONAL_TRANSFER`

P32 showed that matching a downstream functional margin effect globally can
produce a dense low-amplitude intervention that is safe in magnitude but
harmful to native ranking. The P33 hypothesis is that the same signed
functional effect is useful only where the frozen teacher requests a
meaningful intervention. A bounded, source-only, continuous actionability
weight will suppress dense low-actionability micro-corrections while retaining
the signed effect on active locations.

The claim is about *when and where to intervene*, not raw teacher-vector
direction, cosine, sign agreement, Pearson, Spearman, q99, or any other
internal metric. The native/zero-adapter detector is a mandatory control. If
native is non-inferior under the locked criteria, P33 concludes that this
intervention is unnecessary for the declared scope.

## 2. Frozen mechanism and exact equation

For a scalar 9×9 map, the existing Industrial deployment transform is:

```text
D(x) = Bilinear_37_to_518(
         GaussianBlur_7,sigma=1(Bilinear_9_to_37(x)))
```

Both resizes use `align_corners=True`. For student stage residuals `s[g]` and
teacher region target `t`:

```text
E_s = D(mean_g(s[g]))
E_t = D(t)
C   = 4.960109710693359
w   = stop_gradient(clamp(abs(E_t) / C, min=0, max=1))
```

`C` is the inherited repository correction scale from the frozen P29/P30R1
contract. It is not a new tuned scalar and is not estimated from held
outcomes. The sole P33 objective is:

```text
L_P33 = mean(
    w * SmoothL1(E_s, stop_gradient(E_t), beta=1.0, reduction="none"))
```

The teacher effect is detached before both target use and weight derivation.
The weight is pixelwise, continuous, bounded in `[0,1]`, and training-only.
The signed teacher effect remains the target wherever `w` is nonzero; P33
does not replace the target by `w*E_t` and does not impose a null target on
abstaining pixels. Exact zero teacher effect gives zero weight and zero loss;
near-zero effect gives a proportionally small weight.

Tensor contract:

```text
student_region: [3,B,9,9] float32
teacher_region: [B,9,9] float32
student_effect: [B,518,518] float32
teacher_effect: [B,518,518] float32
weight:         [B,518,518] float32, detached, in [0,1]
```

The residual deployment remains the existing symmetric two-class correction:

```text
normal_logit   += -r / 2
abnormal_logit +=  r / 2
m_candidate    = m_native + E_s
p_candidate    = sigmoid(m_candidate)
```

The student is not self-normalized. There is no hard threshold, learned gate,
classifier, category parameter, ranking loss, sign loss, sparsity loss,
normal-region loss, calibration term, auxiliary objective, or target
shrinkage. Objective count is exactly one.

## 3. Frozen architecture, data, and optimization

The future locked attempt must use:

- the canonical Phase2B parent, image size `518`, three stages, and unchanged
  `RegionResidualAdapter`;
- the frozen Tier-A/Tier-B cache and its existing provenance checks;
- the locked LOCO candle fold: 1,962 fit records and 200 held records;
- 20 epochs, batch size 1, seed 0, canonical FP32, and the existing
  deterministic policy;
- AdamW, learning rate `0.001`, betas `(0.9,0.999)`, epsilon `1e-8`, weight
  decay `0.01`, and `amsgrad=False`;
- SmoothL1 beta `1.0`, inherited from the frozen objective convention;
- no new learnable parameters, no new tuned scalar, no category-specific
  parameter, no schedule/optimizer change, and no parameter sweep;
- mandatory native/zero-adapter, frozen P30R1, and frozen P32 comparators;
- exactly one scientific attempt, with no automatic rerun or fallback method.

| item | locked value |
|---|---:|
| scientific objective count | 1 |
| new tuned scalar hyperparameters | 0 |
| inherited correction scale `C` | `4.960109710693359` |
| SmoothL1 beta | inherited `1.0` |
| new learnable parameters | 0 |
| teacher detach | yes |
| student self-normalization | no |
| category-specific parameters | 0 |
| teacher at inference | forbidden |
| incremental inference overhead | 0% |

## 4. Allowed and forbidden data

Allowed before prediction freeze:

- locked candle fit/source samples;
- frozen Tier-A segmentation features and native logits;
- frozen Tier-B teacher region targets;
- immutable metadata required for source identity and cache validation;
- deterministic synthetic tensors and source-only distribution summaries.

Forbidden before prediction freeze:

- held GT labels or held masks;
- held pAP, pAUROC, ranking, score, or any other outcome;
- any held-derived threshold, weight, coefficient, or formulation choice;
- any new source or held cache build;
- any new CLIP, Phase2B, or teacher neural forward;
- any category-specific actionability rule;
- any candidate-selection loop using held results.

Held labels/masks may be read only after candidate and control prediction
artifacts are frozen, for the predeclared final outcomes and descriptive
post-freeze diagnostics. They may not be used to tune or alter P33.

## 5. Forward, cache, and inference contract

Future cached training may run one adapter/student forward per fit batch using
cached segmentation features. The objective may transform the cached student
residual and cached teacher region with the fixed deployment operator. It may
not run a new CLIP, Phase2B, or teacher neural forward. The teacher has no
inference-time role.

P33 adds no inference branch: the weight is not computed at inference and the
deployed candidate uses the existing adapter/scoring path. Incremental
inference overhead relative to the existing adapter path is locked at `0%`.

## 6. Preflight and engineering gates

Before a scientific Stage 2 marker could be created, all gates below must
pass:

1. the frozen equation, deployment operator, tensor shapes, operator rank,
   and conditioning agree with the repository contract;
2. symbolic checks establish `0 <= w <= 1`, finite zero/near-zero behavior,
   retained signed target scale, and no student normalization;
3. deterministic synthetic tests cover exact zero, near zero, `0.01×`,
   `0.1×`, `1×`, `10×`, `100×`, sign reversal, 1% sparse actionable support,
   heavy-tail corruption, mixed-scale batches, one extreme outlier, all
   abstain, and all/high-confidence active cases;
4. every synthetic case has finite loss and gradients, bounded SmoothL1
   influence, no single-sample domination, and the expected weight/update
   semantics;
5. the frozen source-only audit shows bounded weights, no held dependence,
   no category-specific rule, and no source tail requiring a new tuned
   constant;
6. readable reference and production implementations agree in FP32 within
   explicit output/loss/gradient tolerances;
7. the exact cached CLI-to-checkpoint smoke reads no held data, performs no
   new neural forward or cache build, keeps teacher/frozen parameters fixed,
   produces finite gradients, changes the student, saves a checkpoint, and
   strictly reloads it;
8. the 5-step and warmed 40-step engineering profiles pass, with objective
   and input/cache time reported separately; preferred end-to-end and hard
   unexplained overhead gates are `≤10%` and `≤15%` against the closest
   cached baseline;
9. peak memory growth is `≤10%` unless a fixed, documented tensor accounts
   for it; and
10. native, controls, and candidate use identical locked data identity and
    scoring paths.

Failure of a preflight or engineering gate stops P33 and does not authorize
a scientific run.

## 7. Future scientific Stage 2 criteria

The primary outcome is candidate pAP and the secondary outcome is candidate
pAUROC. The locked native non-inferiority thresholds are the frozen P31
native candle references:

```text
pAP    >= 0.5141403049313743
pAUROC >= 0.9806671435137679
```

The candidate must also satisfy the inherited health/safety gates:

```text
global residual absolute q99      <= 8.643353872299194
normal score/effective-margin q99 <= 0.0010011587851122385
nonfinite loss count              = 0
nonfinite gradient count          = 0
```

Native/zero-adapter, frozen P30R1, and frozen P32 are reported as controls or
comparators. Raw residual cosine, sign, Pearson, Spearman, support fraction,
and actionability-weight distributions are diagnostics; none is an
optimization target or a substitute for pAP/pAUROC. A reduction in dense
micro-correction is a mechanism diagnostic, not permission to relax the
locked outcome gates.

If either native non-inferiority endpoint or any health, safety, provenance,
or data-access gate fails, the hypothesis is falsified for the locked candle
scope. If all gates pass, the result supports the declared selective
functional-transfer claim for that scope only; it makes no cross-category
claim.

## 8. Scientific stop and no-rerun policy

Stop without interpretation for any data-access, provenance, forbidden-read,
model-forward, cache, shape, numerical, checkpoint, or runtime violation.

There is exactly one future scientific P33 Stage 2 attempt. A failed or
ambiguous result does not allow changing `C`, beta, normalization, target,
weight formula, threshold, optimizer, schedule, seed, class, sample subset,
loss, architecture, or gates. A repeat is allowed only to repair a documented
integrity failure with identical frozen inputs and code; that repair is not a
new scientific attempt. Any scientific semantic change requires a new
protocol and preregistration.

## 9. Authorization boundary and counts at freeze

This preregistration creates no scientific UUID, execution marker, or attempt.
It authorizes no training or scoring by itself. Engineering-only optimizer
steps, if performed after this freeze, are not scientific Stage 2 attempts
and must use source/fit cache only.

```text
new scientific Stage 2 attempts = 0
new Stage 3 attempts             = 0
full runs                        = 0
held-result tuning iterations    = 0
new CLIP forwards                = 0
new Phase2B forwards             = 0
new teacher forwards             = 0
cache rebuilds                   = 0
scientific UUIDs                 = 0
execution markers               = 0
```

`P33_PREREGISTRATION_FROZEN`
