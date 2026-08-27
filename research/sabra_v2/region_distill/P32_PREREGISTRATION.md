# P32 Preregistration — Functional Margin Effect

Status: `P32_PREREGISTRATION_FROZEN`

This is the authoritative preregistration for one possible future P32
scientific Stage 2 comparison. It is not an execution marker and does not
authorize a scientific run. The byte-level SHA-256 of this file is recorded in
the companion JSON and `P32_PREREGISTRATION_SHA256.txt`.

## 1. Scientific hypothesis

`SELECTED_P32_HYPOTHESIS = FUNCTIONAL_MARGIN_EFFECT`

When raw teacher correction direction is not the causal downstream quantity, a
student trained to match the teacher's exact native-relative deployed margin
effect can transfer useful intervention without being constrained to imitate
the teacher residual vector in raw 243-coordinate space.

The native/zero-adapter detector is a mandatory control. The hypothesis does
not assume that teacher intervention is beneficial. If native is non-inferior
under the locked criteria, the conclusion is that learned intervention is
unnecessary for this scope.

## 2. Frozen mechanism and equation

The frozen repository deployment transform for one scalar 9×9 map is:

```text
T(x) = Interpolate_37_to_518(
         GaussianBlur_7,sigma=1(Interpolate_9_to_37(x)))
```

Both interpolations are bilinear with `align_corners=True`; the Gaussian
kernel and sigma are the existing Industrial deployment values. For student
stage residuals `s[g]` and a teacher target `t`, the deployed margin effects
are:

```text
Delta_m_student = T(mean_g(s[g]))
Delta_m_teacher = T(t)
```

The symmetric two-class correction is fixed as:

```text
normal_logit   += -r / 2
abnormal_logit +=  r / 2
```

Consequently, with the frozen native deployed margin `m_native`,

```text
m_student = m_native + Delta_m_student
p_student = sigmoid(m_student)
```

The native term cancels from the pre-softmax margin effect. The only scientific
objective is:

```text
L_FME = mean SmoothL1(
    Delta_m_student,
    stop_gradient(Delta_m_teacher),
    beta=1.0,
    reduction="mean")
```

Tensor contract:

```text
student_region: [3,B,9,9] float32
teacher_region: [B,9,9] float32
student_effect: [B,518,518] float32
teacher_effect: [B,518,518] float32
```

The teacher is detached and the student is not self-normalized. There is no
cosine, sign, Pearson, Spearman, ranking, sparsity, normal-pixel,
calibration, segmentation, feature, gate, or auxiliary term. `beta=1.0` is
inherited from the frozen P30R1 robust-regression convention; it is not tuned
for P32.

The fixed operator on the frozen geometry has matrix shape `[268324,81]`, rank
`81`, and condition number `6.140363502015776`. P32 therefore tests the exact
deployed margin effect. It does not claim a nullspace or radial
non-identifiability that the operator analysis does not support.

## 3. Frozen architecture, data, and optimization

The following are immutable for the future locked attempt:

- canonical Phase2B parent, prompt, image size `518`, three stages, and
  symmetric margin construction;
- unchanged `RegionResidualAdapter`, including zero-output initialization;
- frozen Tier-A/Tier-B cache root and provenance;
- teacher target semantics and source-only teacher cache;
- one locked LOCO candle fold: 1,962 fit records and 200 held records;
- 20 epochs, batch size 1, seed 0, canonical FP32, and the existing
  deterministic policy;
- AdamW with learning rate `0.001`, betas `(0.9,0.999)`, epsilon `1e-8`,
  weight decay `0.01`, and `amsgrad=False`;
- no new scalar hyperparameters, category-specific parameters, schedule
  change, or optimizer change;
- mandatory native/zero-adapter control and frozen P30R1 comparator;
- fixed score postprocessing and existing pAP/pAUROC implementation;
- exactly one scientific attempt, with no automatic rerun or fallback method.

Objective contract:

| item | locked value |
|---|---:|
| scientific objective count | 1 |
| new tuned scalar hyperparameters | 0 |
| SmoothL1 beta | inherited `1.0` |
| new learnable parameters | 0 |
| teacher detach | yes |
| student self-normalization | no |
| category-specific parameters | 0 |
| teacher at inference | forbidden |
| incremental inference overhead | 0% |

## 4. Allowed and forbidden data

Allowed during future cached training:

- VisA fit/source samples in the locked candle fold;
- frozen Tier-A segmentation features and native logits;
- frozen Tier-B teacher region targets;
- immutable metadata required for source identity and cache validation.

Forbidden before prediction freeze:

- held GT labels or held masks;
- held outcome metrics or any held-derived threshold, weight, or coefficient;
- any new source or held cache build;
- any new CLIP, Phase2B, or teacher neural forward;
- any category-specific actionability rule.

Held labels and masks may be read only after the candidate and native-control
prediction artifacts are frozen, solely for the predeclared final outcomes and
descriptive post-freeze diagnostics.

## 5. Allowed forwards and deployment cost

Future cached training may execute one adapter/student forward per fit batch
using cached inputs. It may not execute a new CLIP or Phase2B forward and may
not execute a teacher neural forward. The functional objective is
training-only.

At inference, P32 uses the existing adapter and deployment operator. There is
no teacher, extra model, gate, ranking branch, or iterative refinement.
Incremental inference overhead relative to the existing adapter path is
locked at `0%`; the already-existing adapter cost relative to native is not
attributed to P32.

## 6. Controls and preflight gates

The future comparison must include:

1. native/zero-adapter output;
2. the frozen P30R1 output comparator;
3. the P32 functional-margin-effect candidate.

Before a scientific Stage 2 marker could ever be created, all of these gates
must pass:

1. frozen equation and repository deployment operator agree exactly;
2. fixed operator rank, conditioning, and tensor shapes remain unchanged;
3. deterministic synthetic adversarial suite is finite;
4. exact zero has zero loss/gradient, zero-teacher nonzero-student has a
   finite restoring gradient, and near-zero targets have no NaN/Inf;
5. `0.01×`, `0.1×`, `1×`, `10×`, and `100×` scale cases have expected radial
   ordering and update direction;
6. sign reversal, 1% sparse corruption, heavy-tail corruption, mixed-scale
   batches, one extreme outlier, all-null, and high-effect cases remain finite
   with bounded SmoothL1 influence;
7. source-only support, scale, tail, and category-dispersion audit passes
   without category-specific tuning;
8. production/reference objective output and student gradients agree within
   the recorded FP32 tolerance;
9. cached production smoke confirms teacher/frozen tensors are unchanged, the
   student update is nonzero, no held data is read, and no cache or model
   recomputation occurs;
10. training median/end-to-end overhead is ≤10% preferred and ≤15% maximum
    versus the closest cached P30R1 path, with objective and input/cache costs
    reported separately; unexplained >15% is an engineering stop;
11. peak memory growth is ≤10% unless a documented fixed tensor explains it;
12. native and candidate use identical locked data identity and scoring paths.

Failure of a preflight or engineering gate stops P32 and does not authorize a
scientific run.

## 7. Future scientific Stage 2 criteria

The outcome comparison is fixed before any future held result. The primary
criterion is candidate pAP and the secondary criterion is candidate pAUROC.
The native non-inferiority thresholds are the frozen P31 native candle
references:

```text
pAP    >= 0.5141403049313743
pAUROC >= 0.9806671435137679
```

The candidate must also satisfy the inherited safety and health criteria:

```text
global residual absolute q99      <= 8.643353872299194
normal score/effective-margin q99 <= 0.0010011587851122385
nonfinite loss count              = 0
nonfinite gradient count          = 0
```

The frozen P30R1 comparator is reported. Raw residual cosine, sign,
Pearson, Spearman, and candidate-versus-teacher ranking are diagnostics only;
they are neither optimization targets nor pass gates. The declared mechanism
is the deployed functional effect, not a metric-rescue criterion.

If either native non-inferiority endpoint or any health, safety, or provenance
gate fails, the hypothesis is falsified for the locked candle scope and P32
stops. If all gates pass, the result supports retaining the functional-effect
hypothesis for the declared scope only; it makes no cross-category claim.

## 8. Scientific stop and rerun policy

Stop without interpretation for any data-access, provenance, forbidden-read,
model-forward, cache, shape, numerical, checkpoint, or runtime violation.

There is exactly one future scientific Stage 2 attempt. A failed or ambiguous
result does not allow a changed beta, normalization, threshold, optimizer,
schedule, seed, class, sample subset, loss, ranking rule, gate, or
architecture. A repeat is allowed only to repair a documented integrity
failure with identical frozen inputs and code; such a repair is not a new
scientific attempt. Any scientific semantic change requires a new protocol and
preregistration.

## 9. Authorization boundary and counts at freeze

This preregistration creates no scientific UUID, execution marker, or future
experiment attempt. It authorizes no training or scoring by itself.

```text
new scientific Stage 2 attempts = 0
new Stage 3 attempts             = 0
full runs                        = 0
held-result tuning iterations    = 0
new CLIP forwards                = 0
new Phase2B forwards             = 0
cache rebuilds                   = 0
optimizer steps                  = 0
```

`P32_PREREGISTRATION_FROZEN`
