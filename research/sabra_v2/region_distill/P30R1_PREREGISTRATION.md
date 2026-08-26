# P30R1 preregistration — teacher-relative radial stabilization

Status: design-only preregistration, frozen 2026-08-26. No P30R1 scientific
training, Stage 2 run, Stage 3 run, full run, or execution marker was created
in this phase.

## Hypothesis and scope

P30's primary blind spot was radial non-identifiability under a self-normalized
direction-only loss. Its candle Stage 2 improved directional transfer but
failed through `HEAVY_TAIL_RESIDUAL_SCALE_INSTABILITY`, followed by
`LOGIT / ANOMALY_SCORE_SATURATION`: residual absolute q99 was 25.9297989464
versus 4.3216769361 for P29, while normal-score q99 shift was 0.9986907840
versus 0.0000011588. This was a residual-tail failure, not global mean
magnitude inflation.

P30 also excluded exact-zero teacher samples from its directional valid set.
P30R1 tests the minimum correction: one robust regression objective, a
teacher-only radial denominator, and active zero targets. The descriptive name
TSNRD is not a novelty claim. The current description is:

> a mechanism-driven single-objective teacher-relative radial stabilization
> for signed residual distillation in a scale-sensitive anomaly-score
> deployment path.

`PASS_TO_IMPLEMENTATION` from the preflight means only that this frozen
formulation survived cheap falsification. It is not authorization to execute
Stage 2.

## Exact formulation

For each sample, the student is `[3,9,9]` and the cached teacher is `[9,9]`.
The teacher is detached, staged by broadcasting across the three stages, and
both tensors are flattened to 243 coordinates:

```text
t_bar = t / C
s_bar = s / C
a_t   = stop_gradient(sqrt(mean(t_bar^2) + eps^2))
z_t   = t_bar / a_t
z_s   = s_bar / a_t
L     = F.smooth_l1_loss(z_s, z_t, beta=1.0, reduction="mean")
```

Here `C = 4.960109710693359` is the inherited global correction scale and
`eps = 0.01` is the inherited P30 numerical epsilon. The `mean` in `a_t` is
per sample over 243 coordinates; the final loss is the mean over `B * 243`
coordinates. The same teacher-derived `a_t` is used for both sides. The
student is never self-normalized. Exact-zero teacher samples have `a_t = eps`,
`z_t = 0`, and remain in the loss.

This is exactly one scientific objective. There are no cosine, sign, normal,
additional magnitude, ranking, feature, segmentation, calibration,
class-specific, or learned normalization terms. The canonical formulation
text is hashed as:

```text
290aae42e04d9faae5a10b929eb58aa0da066b5dbd248b3fee40f20e9094781c
```

The machine-readable formulation, mathematical cases, and all synthetic
measurements are in
`research/sabra_v2/region_distill/P30R1_PREFLIGHT_FALSIFICATION.json`.

## Mathematical consequences A–J

Let `t = alpha*u`, `s = beta*v`, `||u||_2 = ||v||_2 = 1`, and let
`q_u = sqrt(mean(u^2))`. Then

```text
a_t = sqrt((alpha/C)^2 * q_u^2 + eps^2)
delta = z_s - z_t = (beta*v - alpha*u) / (C*a_t)
dL/ds_j = psi(delta_j) / (B*243*C*a_t)
```

where `psi(x) = x` for `|x| < 1` and `sign(x)` otherwise. Therefore
`|dL/ds_j| <= 1/(B*243*C*eps)`. At batch size one the fixed analytic bounds
are max absolute gradient `0.0829664378` and per-sample L2 `1.2933187702`.

- A (`v=u`, `beta=0.1*alpha`): positive radial loss; the gradient points
  toward increasing `beta`, and the error is not discarded as in P30.
- B (`v=u`, `beta=alpha`): `z_s=z_t`, so loss and gradient are exactly zero;
  this is the unique radial minimum for fixed teacher and direction.
- C (`v=u`, `beta=10*alpha`): large positive loss; the gradient points toward
  decreasing `beta`, with bounded SmoothL1 tail derivative.
- D (`v=u`, `beta=100*alpha`): still a large, approximately linear tail loss;
  it cannot masquerade as a low-loss directional match and has no loss-driven
  explosion.
- E (`v=-u`, `beta=alpha`): positive sign/directional residual; the gradient
  points from `-u` toward `+u` and does not vanish through self-normalization.
- F (correct direction, catastrophic magnitude): radial error is explicitly
  identified; the restoring gradient is bounded rather than quadratic-tail
  explosive.
- G (`t=0`, `s!=0`): finite positive loss with `a_t=eps`; the gradient points
  toward zero, restoring the force P30 removed.
- H (`||t||` near zero): epsilon prevents a singular denominator. This has a
  known floor bias and bounded stronger weighting risk, so it is tested at
  `1e-8`, `1e-6`, and `1e-4` without tuning epsilon.
- I (large teacher): `a_t` tracks teacher RMS, so raw gradient weight falls as
  teacher radius grows; large raw targets do not dominate solely by size.
- J (mixed scales `0, .01, .1, 1, 10, 100`): every sample retains its own
  radial minimum. Small-target domination remains the principal risk, bounded
  and tested by the fixed mixed-batch max/median gradient limit of 100.

For `v=u`, SmoothL1 is strictly minimized at `beta=alpha` for a nonzero
teacher: `0.1x`, `10x`, and `100x` are not equivalent. This is the required
difference from P30 and the single residual regression is the required
difference from P29's `value + sign + normal` objective.

## Frozen scalars, optimizer, and architecture

There are zero new tuned hyperparameters. The complete scalar contract is:

- inherited global `C = 4.960109710693359`;
- inherited `eps = 0.01`;
- inherited SmoothL1 `beta = 1.0`, `reduction = "mean"`;
- unchanged AdamW learning rate `0.001`, betas `(0.9, 0.999)`, epsilon
  `1e-8`, weight decay `0.01`, and `amsgrad = false`;
- unchanged 20 epochs, batch size 1, seed 0, canonical FP32, and deterministic
  algorithms with `warn_only`.

The only trainable component remains the unchanged `RegionResidualAdapter`
with its unchanged zero-output-head initialization. The frozen Tier-A/Tier-B
cache, VisA development universe, 12 fixed leave-one-class-out folds, teacher
semantics, source-only cache provenance, P26 symmetric-margin deployment,
logit construction, and anomaly scoring remain unchanged. P30R1 normalization
exists only inside training loss. There is no teacher, new model, new head,
iterative pass, or category-specific calibration at inference; target inference
overhead is 0%.

SmoothL1 is frozen because it is already the project convention in P29, has a
fixed beta with no new sweep, has a linear tail with bounded derivative, and is
less tail-sensitive than L2 while retaining a simple residual interpretation.
L1 and L2 are recorded as mathematical comparisons only; no loss sweep is
authorized.

## Frozen Stage 2 identity and gates

The later qualification identity is the same fixed one-class `candle` LOCO
fold used by P30: 1,962 fit records, 200 held records, 20 epochs, batch size 1,
and 39,240 expected optimizer steps. The class was fixed before any P30R1
scientific result. Predictions must be frozen before scoring; held GT and held
masks are prohibited before that freeze.

The following gates are frozen now, using only the already-frozen native,
P29, and P30 engineering references. All displayed metrics must also be
finite.

| Gate | Fixed Stage 2 criterion |
|---|---:|
| pAP | `>= 0.4641403049313743` (`native candle pAP - 0.05`) |
| pAUROC | `>= 0.9306671435137679` (`native candle pAUROC - 0.05`) |
| staged directional cosine | `>= 0.6985491737886378` (`P29 - 0.01`) |
| sign agreement | `>= 0.5554938271604938` (`P29 - 0.01`) |
| global absolute residual q99 | `<= 8.643353872299194` (`2 * P29 q99`) |
| normal score/effective-margin q99 shift | `<= 0.0010011587851122385` (`P29 q99 + 0.001`) |
| nonfinite gradient count | `0` throughout |
| nonfinite loss count | `0` throughout |
| median training step | `<= 0.007889637351036071` seconds, 15% over frozen P30 |
| training wall time | `<= 1447.5493008233607` seconds, 15% over frozen P30 |

The preferred runtime envelope is 10% over P30; over 15% is an engineering
stop unless a preregistered measurement artifact explains it. Teacher gradient
must remain `None/zero`, no new CLIP or Phase2B forwards may occur, and cache
and prediction provenance must match the frozen manifests and checksums.

Every metric, gradient, provenance, and runtime criterion must pass before the
protocol can advance. No post-result threshold, epsilon, beta, loss weight, or
class choice change is permitted.

## Stage dependencies and stop rules

Stage 0 is the isolated static/preflight gate. A later prompt may run at most a
one-step engineering smoke after Stage 0 passes; it is not a scientific
result. The smoke must confirm finite forward/backward/update, changed student
parameters, unchanged teacher, zero held reads before freeze, zero new CLIP and
Phase2B forwards, and cache compatibility.

Stage 2 may start only after Stage 0 and Stage 1 pass. Stage 3 is fixed to
`candle`, `chewinggum`, `macaroni2`, and `pcb3`, and the full stage is fixed to
all 12 classes in canonical order. Both remain impossible unless Stage 2 has a
recorded PASS for every gate above and a later explicit execution prompt is
provided. No execution marker exists now; the full marker may be created only
immediately before a later authorized full training attempt.

Stop and return to research if scale invariance persists, 10x/100x is weakly
penalized, zero-teacher gradients vanish, near-zero gradients become unstable,
the source radial distribution fails its fixed bounds, heavy-tail corruption
is not detected, any tuning or hidden objective appears, any forbidden read
occurs, or any Stage 2 metric/provenance/runtime gate fails. Do not invent an
automatic P30R2.

The machine-readable companion freezes the same contract and is hashed in
`P30R1_PREREGISTRATION_SHA256.txt`.
