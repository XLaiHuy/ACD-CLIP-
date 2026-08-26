# P31 Preregistration — Native / Zero-Adapter Control

Status: `FROZEN_AUTHORITATIVE_PREREGISTRATION`

Protocol identifier: `P31`

This protocol is evaluation-only. It is not a training authorization and does
not authorize a new scientific Stage 2 run in the current engineering phase.

## 1. Scientific hypothesis

`P31_NATIVE_ZERO_ADAPTER_CONTROL`

On the locked SABRA one-class comparison, the frozen native detector is
non-inferior to the P30R1 teacher-residual intervention. Raw teacher residual
imitation is therefore not necessary as a default SABRA component unless a
separate future preregistered test demonstrates a downstream gain.

The claim is scoped to the frozen comparison and named data. It is not a claim
that the native detector is optimal for every class or dataset.

## 2. Causal mechanism and exact formulation

P30R1 forensic mechanism:

```text
PRIMARY_MECHANISM   = TEACHER_DIRECTION_NOT_CAUSAL
SECONDARY_MECHANISM = SPARSE_SELECTIVE_CORRECTION
```

P31 tests the causal necessity of intervention by setting it to identity:

```text
R_P31(x)       = 0
L_P31(x)       = L_native(x)
A_P31(x)       = A_native(x)
Delta_P31(x)   = A_P31(x) - A_native(x) = 0
```

Here `R_P31` is conceptual only. No adapter is instantiated and no residual
is passed through a model. The implementation takes a finite native output
array `N` and returns `copy(N)` exactly. The diagnostic delta is computed as
`copy(N) - N` and must be exactly zero for equal in-memory values.

No teacher direction, sign, norm, cosine, Pearson, Spearman, sparsity,
confidence, score threshold, or teacher scale is optimized or used to select a
method.

## 3. Frozen tensor and output contract

- Cached native logit shape, when inspected, is `[3, B, 1369, 2]`.
- Stored native anomaly-map shape, when inspected, is `[B, 518, 518]`.
- The identity function accepts any finite array with a non-empty shape and
  returns an immutable independent copy with identical shape, dtype, and
  values.
- Non-finite input is an engineering/data-access failure, not a value to
  sanitize or clip.
- The output must not depend on a teacher residual, source scale, mask,
  confidence, category, or held result.

## 4. Frozen model, architecture, and data provenance

- Native CLIP/P26/Phase2B provenance, checkpoint identities, and deployment
  operator remain unchanged.
- No `RegionResidualAdapter` is constructed or called by P31.
- No P29, P30, or P30R1 evidence, checkpoint, implementation, or inference
  path is modified.
- The read-only source audit may inspect `/workspace/p27r1_cache_v1` Tier-A
  native logits and Tier-B teacher regions to verify finite source statistics.
- The scientific comparison uses the already frozen native/zero-adapter and
  P30R1 outputs recorded by the P30R1 forensic. This engineering phase does
  not recompute held predictions or metrics.

## 5. Allowed and forbidden data

Allowed:

- the P30R1 forensic report and JSON with status `FORENSIC_COMPLETE`;
- immutable native and P30R1 output artifacts and their provenance metadata;
- source-only Tier-A/Tier-B cache arrays for finite-value and shape audits;
- held labels only after this preregistration is locked and only for the
  final, predeclared pAP/pAUROC calculation of a future scientific comparison.

Forbidden:

- held labels or held masks for method selection, formulation design,
  threshold selection, coefficient selection, or rerun decisions;
- any new class, dataset, 12-class expansion, or new scientific held output;
- any cache rebuild, source-mask read, teacher reconstruction, or teacher
  forward;
- any CLIP, Phase2B, adapter, native-model, or deployment-model forward in
  this engineering phase;
- any learned gate, residual, loss, objective, auxiliary network, category
  parameter, confidence rule, or teacher-at-inference path;
- changing the output metric or postprocessing after seeing a result.

## 6. Allowed model forwards and execution counts

P31 allows exactly zero new model forwards in this phase:

```text
new CLIP forwards       = 0
new Phase2B forwards    = 0
new teacher forwards    = 0
adapter forwards        = 0
optimizer steps         = 0
cache rebuilds          = 0
```

The implementation is an offline pure-array/metric contract. If a missing or
corrupt artifact would require a model forward, stop with an engineering/data
access failure and do not recompute it under this protocol.

## 7. Objectives, parameters, constants, and seed policy

| Item | Frozen value |
|---|---:|
| New training objectives | `0` |
| New learnable parameters | `0` |
| New tuned hyperparameters | `0` |
| Residual | analytic exact zero |
| Non-inferiority margin | `0.0` absolute metric units |
| Native reconstruction audit tolerance | inherited `2e-5` maximum absolute error |
| Precision for cached arrays | FP32 as stored; no model computation |
| Optimizer | none |
| Schedule | none |
| Random seed | inherited `0` if a host framework requires one; no RNG operation or sampling |

The reconstruction tolerance is an engineering parity bound inherited from
the existing frozen replay diagnostics, not a scientific tuning parameter.
The zero metric margin is fixed before any future comparison and is not
estimated from held labels.

## 8. Null and comparator controls

- **P31 treatment/control:** native output with exact zero intervention.
- **Frozen comparator:** existing P30R1 teacher-residual output on the same
  locked sample set.
- **No learned baseline:** no new adapter, loss, checkpoint, or output branch
  is created.

The purpose is to distinguish useful correction from do-no-harm preservation.
If native is non-inferior, a further teacher-imitation method is not warranted
by this protocol. If native is inferior, that falsifies the P31 control for the
scoped comparison but does not authorize a follow-up method automatically.

## 9. Synthetic/preflight gates

The required deterministic suite is recorded in
`P31_PREFLIGHT_FALSIFICATION.json`. It includes:

- exact zero and all-null/no-intervention;
- near-zero;
- normal scale, `0.01x`, `0.1x`, `1x`, `10x`, and `100x`;
- sign flip/opposite direction;
- 1% sparse corruption;
- finite heavy-tail corruption;
- mixed-scale batch;
- one extreme outlier sample;
- high-confidence intervention.

Every case must satisfy:

```text
finite output                 = true
output delta                  = exactly zero
objective count               = 0
loss                          = undefined/null
student gradient              = exactly zero
expected update               = NO_UPDATE
one-sample dominance          = false
```

The source-only audit must report finite Tier-A/Tier-B arrays, zero held
label/mask reads, and zero model forwards. A failure is `ENGINEERING_STOP` and
does not justify changing the formulation.

## 10. Engineering gates

Before any future evaluation-only invocation:

1. Verify the forensic status, frozen input identities, shapes, dtypes, and
   hashes.
2. Verify native reconstruction against the existing deterministic operator
   within the inherited `2e-5` maximum absolute-error bound.
3. Verify native identity output is an independent exact copy and has zero
   delta.
4. Verify the invocation records zero model forwards, optimizer steps, cache
   rebuilds, and scientific markers.
5. Verify no existing P29/P30/P30R1 file or checkpoint changed.
6. Run the offline production smoke and speed profiles using synthetic or
   source-only arrays only.

## 11. Future scientific Stage 2 criteria

P31 Stage 2 is an evaluation-only paired comparison. It is not executed in
this phase and must use the locked cached native and P30R1 outputs on the same
sample set and postprocessing path.

Define:

```text
Delta_pAP    = pAP(native zero-adapter) - pAP(P30R1)
Delta_pAUROC = pAUROC(native zero-adapter) - pAUROC(P30R1)
```

Primary endpoint: pAP. Secondary endpoint: pAUROC.

The P31 hypothesis is supported for the scoped comparison when:

```text
Delta_pAP >= 0 and Delta_pAUROC >= 0
```

The hypothesis is falsified for the scoped comparison when either difference
is negative. No statistical margin, threshold, score transformation, or
sample subset may be introduced after seeing the outcome.

## 12. Scientific stop rules

Stop without a scientific conclusion for any provenance mismatch, missing or
non-finite input, invalid metric, forbidden data read, new model forward,
cache rebuild, or scientific marker.

If both differences are nonnegative, record native-control support and stop;
do not create a learned method merely to continue protocol numbering.

If either difference is negative, record control falsification and stop. A
future downstream-logit-effect or rank/margin hypothesis would require a new
research decision, new preregistration, engineering qualification, and a new
protocol identifier or explicitly amended protocol.

## 13. No-rerun rule

There is one locked comparison. An unfavorable result does not permit reruns
with changed thresholds, sample subsets, coefficients, objectives, losses,
postprocessing, precision, or data. A repeat is allowed only to resolve a
documented integrity/reproducibility failure using identical frozen inputs and
code. A changed scientific question requires a new preregistration.

Held labels cannot be revisited to select a replacement method after the
comparison. A failed control is a result, not an automatic queue.

## 14. Runtime and memory gates

The implementation must be O(N) in the number of values copied or validated,
with no pairwise operation, per-pixel Python loop, model forward, teacher
branch, or retained computation graph.

```text
training overhead       = 0%
new inference overhead  = 0%
new model memory        = 0
```

The 5-step micro-profile and 40-step warmed profile are engineering profiles
of the offline control operation, not training profiles. They must report
startup, input/validation, copy, delta, total step, and RSS where available.
No batch-size, precision, optimizer, schedule, or scientific data reduction
may be changed for speed.

## 15. Scientific deviation policy

After this file is hashed, any change to the hypothesis, exact identity rule,
tensor contract, tolerance, metric, margin, allowed data, forward policy,
stop rule, or rerun rule is a `PREREGISTRATION_DEVIATION_STOP`.

No autonomous loss, lambda, epsilon, normalization, gate, architecture,
optimizer, schedule, seed, or source-selection change is allowed.

## 16. Terminal authorization

This file freezes a future evaluation-only protocol. It does not create a
scientific UUID, execution marker, final experiment result, trainer, runner,
checkpoint, or new scientific Stage 2 attempt.

`P31_PREREGISTRATION_FROZEN`
