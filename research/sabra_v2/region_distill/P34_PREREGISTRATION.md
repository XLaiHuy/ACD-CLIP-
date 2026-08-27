# P34 Preregistration — Explicit Actionability Target Functional Transfer

Status: `P34_PREREGISTRATION_FROZEN`

This is the authoritative preregistration for one possible future P34
Scientific Stage 2 attempt. It creates no scientific UUID, execution marker,
prediction, or held result. The companion JSON records the SHA-256 of this
Markdown file.

## 1. Scientific hypothesis

`SELECTED_NEXT_HYPOTHESIS = EXPLICIT_ACTIONABILITY_TARGET_FUNCTIONAL_TRANSFER`

P33 used a source-only actionability signal as loss importance:

```text
L_P33 = mean(w * SmoothL1(E_s, E_t; beta=1))
```

At `w=0`, this removes the direct gradient even when the shared adapter has a
nonzero student correction. P34 tests whether explicit target semantics
restore selectivity:

> If the actionability signal defines the desired correction itself, then
> low-actionability locations receive a zero target and a restoring gradient,
> while high-actionability locations retain the functional teacher target.

P34 tests *when and how strongly to intervene*. It does not test teacher
direction fidelity, cosine, sign, Pearson, Spearman, ranking loss, a sparsity
penalty, a learned gate, or a metric repair.

## 2. Frozen formulation

Let `D` be the exact existing deployment transform:

```text
D(x) = Bilinear_37_to_518(
         GaussianBlur_7,sigma=1(Bilinear_9_to_37(x)))
```

Both bilinear operations use `align_corners=True`; Gaussian kernel is `7x7`
with sigma `(1.0,1.0)`. For student residual stages `s[g]` and cached teacher
region target `t`:

```text
E_s = D(mean_g(s[g]))
E_t = D(t)
C   = 4.960109710693359
w   = stop_gradient(clamp(abs(E_t) / C, min=0, max=1))
T   = stop_gradient(w * E_t)
L_P34 = mean(SmoothL1(E_s, T; beta=1.0, reduction="none"))
```

`teacher_region` is detached before `E_t`; `w` and `T` are detached. The
detachment prevents a target-side graph and makes the target immutable during
the student update. `T` is the only new semantic difference from P33.

The exact derivative, for `N` effect pixels, is:

```text
psi_beta(u) = sign(u) * min(abs(u)/beta, 1)
dL_P34/dE_s = psi_1(E_s - w*E_t) / N
```

Thus `w=0` gives a zero target and a zero-seeking gradient whenever `E_s` is
nonzero; `w=1` gives the ordinary P32 functional target. The target is
attenuated continuously for intermediate `w`.

Tensor contract:

```text
student_region: [3,B,9,9] float32
teacher_region: [B,9,9] float32
student_effect: [B,518,518] float32
teacher_effect: [B,518,518] float32
actionability: [B,518,518] float32, detached, [0,1]
target_effect: [B,518,518] float32, detached
```

The student is not self-normalized. Student scale remains identifiable. There
is exactly one objective, no auxiliary term, no new learnable parameter, no
new tuned scalar, no category-specific parameter, and no inference-time P34
calculation.

## 3. Fixed architecture, data, and optimization

The one future scientific attempt, if engineering gates pass, must use:

- canonical Phase2B parent, image size `518`, three stages, and unchanged
  `RegionResidualAdapter`;
- the frozen Tier-A/Tier-B P27 cache and its provenance checks;
- the locked LOCO candle split: 1,962 fit/source records and 200 held records;
- fit/source cache only before prediction freeze;
- 20 epochs, batch size 1, seed `0`, deterministic policy, canonical FP32;
- AdamW with learning rate `0.001`, betas `(0.9, 0.999)`, epsilon `1e-8`,
  weight decay `0.01`, and `amsgrad=False`;
- expected optimizer steps `39240` (`1962 * 20`);
- unchanged schedule, architecture, cache identity, and checkpoint format;
- no parameter sweep, alternate seed, alternate batch size, alternate class,
  or fallback objective.

| item | frozen value |
|---|---:|
| objective count | 1 |
| new tuned scalar hyperparameters | 0 |
| new learnable parameters | 0 |
| inherited correction scale `C` | 4.960109710693359 |
| SmoothL1 beta | 1.0 |
| teacher target detached | yes |
| student self-normalization | no |
| category-specific parameters | 0 |
| teacher at inference | forbidden |
| inference overhead | 0% |

## 4. Allowed and forbidden evidence

Allowed before a future prediction is frozen:

- locked candle fit/source records and frozen cache tensors;
- cache manifests and provenance metadata;
- deterministic synthetic tensors;
- the frozen P34 preflight artifact;
- engineering-only optimizer steps on fit/source cache;
- static code, reference parity, smoke, and speed measurements.

Forbidden before future prediction freeze:

- held GT, masks, pAP, pAUROC, ranking, score, or anomaly enrichment;
- any held-derived threshold, coefficient, gate, or target transform;
- any source/held cache rebuild;
- any new CLIP, Phase2B, or teacher forward;
- any class-specific rule or tuning loop;
- any change to this equation, `C`, beta, optimizer, schedule, seed, or
  architecture.

Held labels and masks may be read only after the future P34 prediction is
written, hashed, and marked `P34_PREDICTION_FROZEN`, solely for the declared
endpoint and descriptive diagnostic calculations.

## 5. Controls and inference semantics

The future report must use identical locked scoring and sample identity for:

- P31/native zero-adapter control;
- frozen P30R1;
- frozen P32;
- frozen P33;
- P34.

P34 target/actionability is training-only. At inference, the existing adapter
and deployment/scoring path are used without a teacher, target, gate, extra
network, or new branch. Incremental inference overhead is locked at `0%`.

## 6. Preflight and engineering gates

Before a future scientific marker may be created, all of these must pass:

1. exact equation, tensor shapes, deployment operator, rank, conditioning,
   detachment, and no-self-normalization checks;
2. algebraic zero-restoring-gradient, zero-optimum, `w=1`, and intermediate
   target tests;
3. deterministic synthetic coverage of zero, near-zero, scales from `0.01x`
   through `100x`, sign reversal, sparse actionable support, heavy tail,
   mixed-scale batch, outlier, all-abstain, and all-active cases;
4. finite loss and gradients, bounded SmoothL1 influence, and no uncontrolled
   single-sample batch domination;
5. source-only support/mass audit with no new tuned constant and no held read;
6. explicit P33-versus-P34 zero-weight regression: P33 gradient is zero while
   P34 gradient is nonzero and points toward zero for `E_s!=0`;
7. readable reference and production parity in FP32 with output, loss, and
   gradient tolerances `atol=1e-6`, `rtol=1e-6`;
8. exact cached CLI-to-checkpoint smoke with finite loss/gradients, changing
   student, unchanged frozen parameters, strict reload, no held read, no new
   neural forward, and no cache rebuild;
9. 5-step and warmed 40-step profiles, with objective and input/cache time
   separated; preferred end-to-end overhead `<=10%` and hard unexplained
   overhead limit `15%` against the closest comparable cached baseline;
10. peak memory growth `<=10%` unless a fixed documented tensor accounts for
    it; and
11. zero P34 scientific UUID/marker before authorization.

The frozen source-only preflight is
[`P34_PREFLIGHT_FALSIFICATION.json`](P34_PREFLIGHT_FALSIFICATION.json),
SHA-256
`059d8b1d0cf7b0cfa4999dd2ec03129443e5e78e5a9ce24ca6401bb482f8494c`.

## 7. Future scientific outcomes and gates

The primary endpoint is pAP. The secondary endpoint is pAUROC. The native
non-inferiority references are frozen from P31:

```text
pAP    >= 0.5141403049313743
pAUROC >= 0.9806671435137679
```

The future candidate must also satisfy:

```text
global residual absolute q99      <= 8.643353872299194
normal score/effective-margin q99 <= 0.0010011587851122385
nonfinite loss count              = 0
nonfinite gradient count          = 0
```

### Mechanism gate

Define the inherited source-derived deployed-effect diagnostic epsilon as
`epsilon=C/100=0.0496010971069336`. For a frozen deployed residual map `R`,
define:

```text
active_fraction = mean(abs(R) > epsilon)
effective_support_fraction = ((sum(abs(R)))^2 / sum(R^2)) / number_of_pixels
gini = Gini(abs(R))
```

The P34 mechanism gate is preregistered as:

```text
active_fraction_P34 < 0.999074074074
effective_support_fraction_P34 < 0.962760408648
gini_P34 > 0.069176234345
```

These are relative comparisons with the frozen failed P33 regime using the
same inherited epsilon. They do not require reproducing P30R1's historical
support percentage and are not optimized targets. Detection gates protect
against the trivial all-zero/native solution; the mechanism gate tests
whether explicit target semantics reduce dense intervention.

All pAP/pAUROC, safety, provenance, data-access, and mechanism gates are
jointly required. A failure is a scientific stop for the locked candle scope;
it does not authorize changing the target or running a variant.

## 8. Prediction freeze and scoring order

The future attempt must follow this order:

1. create exactly one scientific UUID and record commit, preregistration hash,
   cache identities, split, seed, and expected steps;
2. train once on the locked fit/source cache;
3. audit optimizer steps, finite values, student delta, frozen delta, forward
   counts, and cache use;
4. generate candidate and control predictions without reading held labels or
   masks;
5. write and SHA-256 hash the prediction, then mark
   `P34_PREDICTION_FROZEN`;
6. only then read held labels/masks and compute the frozen metrics and
   descriptive diagnostics;
7. apply the gates once, write the audit/report, commit evidence, and stop.

## 9. Scientific stop and no-rerun rule

There is exactly one future P34 Stage 2 attempt. No rerun, alternate seed,
parameter change, target transform, hard gate, sparsity loss, objective,
schedule, optimizer, class, or threshold is permitted after prediction or
result. Engineering integrity repair is not a scientific rerun only if it
uses identical frozen code semantics and inputs; otherwise the protocol is
invalid and must stop.

Any scientific semantic change requires a new protocol and preregistration.

## 10. Authorization boundary and counts at freeze

This preregistration freezes the formulation but authorizes no P34 scientific
Stage 2 attempt. Engineering-only optimizer steps may occur after freeze on
fit/source cache only.

```text
P34 scientific Stage 2 attempts = 0
Stage 3 attempts                = 0
full runs                       = 0
held tuning iterations         = 0
new CLIP forwards              = 0
new Phase2B forwards           = 0
teacher forwards               = 0
cache rebuilds                 = 0
scientific UUIDs               = 0
execution markers              = 0
```

`P34_PREREGISTRATION_FROZEN`
