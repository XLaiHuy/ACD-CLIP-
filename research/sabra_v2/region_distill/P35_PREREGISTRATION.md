# P35 Preregistration — Soft Actionability-Weighted Functional Transfer

Status: `P35_PREREGISTRATION_FROZEN`

Protocol: `P35`. This document freezes one possible future P35 Scientific
Stage 2 attempt. It does not create a scientific UUID and does not authorize
that attempt by itself.

## 1. Scientific hypothesis

P34 reduced intervention density but degraded detection after changing the
desired correction from the full functional teacher effect `E_t` to `wE_t`.
P33 retained `E_t` and improved pAP, but its clamp assigned maximum importance
to `43.8716992%` of source pixels. P35 tests the narrower claim that P33's
benefit is optimization-importance allocation and that removing this hard
source plateau while preserving the complete target can improve transfer.

`SELECTED_P35_HYPOTHESIS = SOFT_ACTIONABILITY_WEIGHTED_FUNCTIONAL_TRANSFER`

The causal question is:

> Does a bounded, monotonic, source-only, non-saturating importance map
> preserve the full functional target while allocating optimization mass more
> usefully than P33's hard clamp?

P35 does not claim that residual sparsity is the objective, does not restore
teacher-vector direction as a separate target, and does not use P34 target
shaping.

## 2. Exact formulation

Let `D` be the frozen deployment transform:

```text
D(x) = Bilinear_37_to_518(GaussianBlur_7_sigma_1(Bilinear_9_to_37(x)))
E_s = D(mean_stage(student_region))
E_t = D(teacher_region)
C = 4.960109710693359
x = abs(stop_gradient(E_t)) / C
w = stop_gradient(tanh(x))
L_P35 = mean(w * SmoothL1(E_s, stop_gradient(E_t), beta=1.0,
                          reduction="none"))
```

The target is the full signed `E_t` for every value of `w`. P35 therefore
differs from P34 exactly in this semantic respect:

```text
P34: target = stop_gradient(w * E_t)
P35: target = stop_gradient(E_t)
```

P35 differs from P33 only in the source-example weight map:

```text
P33: w = clamp(abs(E_t)/C, 0, 1)
P35: w = tanh(abs(E_t)/C)
```

`E_t`, `w`, and the target are detached. The student is not self-normalized.
There is exactly one objective, inherited SmoothL1 beta `1.0`, no auxiliary
term, no sparsity regularizer, no hard threshold, no learned gate, no
category-specific parameter, and no new learnable parameter.

Tensor contract:

| tensor | shape | dtype/gradient semantics |
|---|---|---|
| `student_region` | `[3,B,9,9]` | float32, trainable path |
| `teacher_region` | `[B,9,9]` | float32, cached/frozen |
| `student_effect` | `[B,518,518]` | float32, trainable path |
| `teacher_effect` | `[B,518,518]` | float32, detached |
| `weight` | `[B,518,518]` | float32, detached, `[0,1]` |
| target | `[B,518,518]` | float32, detached, exactly `E_t` |

At `w=0`, P35 intentionally provides zero direct loss gradient, because it is
an importance-weighting experiment. It does not provide P34's restoring zero
target. This is the hypothesis under test, not an omission.

## 3. Frozen architecture and deployment

- Parent: canonical Phase2B.
- Adapter: existing `RegionResidualAdapter`, three stages.
- Deployment image size: `518`.
- Training effect transform: the exact symmetric margin-effect transform
  already used by P32/P33.
- Inference: existing native plus adapter deployment path.
- The source-only weight is training-only and is not computed at inference.
- Inference overhead: `0%`.

## 4. Frozen data and provenance

- Dataset: VisA.
- Protocol split: locked candle LOCO fit/held split.
- Fit records: `1962`.
- Held records: `200`.
- Cache root: `/workspace/p27r1_cache_v1`.
- Training reads only the locked Tier-A/Tier-B fit/source cache and its
  provenance metadata.
- The teacher region comes from the frozen source cache; no teacher model is
  loaded or forwarded.
- The source-only preflight used
  `/workspace/p27r1_cache_v1/tier_b/candle/teacher_region.npy`, excluding
  held-class records by manifest identity.

Allowed before a future prediction freeze: fit/source cache, cache
provenance, frozen inherited artifacts, deterministic synthetic tensors, and
engineering metadata. Forbidden before prediction freeze: held GT, held
masks, held metrics, held-derived thresholds or coefficients, and any new
CLIP, Phase2B, or teacher forward. Cache rebuilds are forbidden.

The P35 preflight and source analysis used zero held reads, zero new CLIP
forwards, zero new Phase2B forwards, zero teacher forwards, and zero cache
rebuilds.

## 5. Frozen optimization schedule

- Epochs: `20`.
- Batch size: `1`.
- Expected optimizer steps: `39,240`.
- Optimizer: AdamW.
- Learning rate: `0.001`.
- Betas: `(0.9, 0.999)`.
- Epsilon: `1e-8`.
- Weight decay: `0.01`.
- AMSGrad: `false`.
- Seed: `0`.
- Precision: float32.
- Schedule change: `false`.

## 6. Preflight and engineering gates

The source/synthetic preflight is
[`P35_PREFLIGHT_FALSIFICATION.json`](P35_PREFLIGHT_FALSIFICATION.json),
SHA-256
`fe9d7bca9e2f089f9cc233abef5ec91d3bc5a835d197c4fa23d0d68ce2725032`.
It must remain `P35_PREFLIGHT_PASS`.

Before a future scientific UUID may be created, all of these must pass:

1. source manifest identity and no held-class leakage;
2. finite, bounded, monotonic `tanh` weighting;
3. full-target identity: target equals detached `E_t`, never `wE_t`;
4. deterministic zero, near-zero, mid-scale, high-scale, sign-reversal,
   heavy-tail, mixed-scale, 90%-low/10%-high, outlier, all-zero, and
   all-high synthetic cases;
5. radial identifiability: loss varies with student scale and student effect
   is not self-normalized;
6. one objective, zero new tuned scalar, zero category-specific state;
7. production/reference FP32 parity;
8. cached one-step forward/backward/optimizer smoke and strict checkpoint
   reload;
9. P34 reporting-schema regression and mock final-report generation;
10. five-step and warmed forty-step engineering profiles are finite and have
    no unexplained overhead above the frozen engineering limit.

Source weight distributions are diagnostics for validating the frozen map;
held residual support and anomaly enrichment are not preflight tuning inputs.

## 7. Future Scientific Stage 2 gates

If separately authorized, exactly one P35 candle Stage 2 attempt must satisfy
all of the following frozen gates:

| gate | frozen requirement |
|---|---:|
| pAP | `>= 0.5141403049313743` |
| pAUROC | `>= 0.9806671435137679` |
| global residual absolute q99 | `<= 8.643353872299194` |
| normal-score q99 shift | `<= 0.0010011587851122385` |
| nonfinite loss count | `== 0` |
| nonfinite gradient count | `== 0` |

The pAP and pAUROC thresholds are the frozen P31/native endpoints. The tail
limits are inherited safety limits from the previously frozen protocol. All
gates are required. The following remain descriptive mechanism diagnostics,
not success gates: residual support fraction, effective support, Gini,
top-10% residual mass, weight quantiles, and anomaly enrichment.

The future report must compare P31/native, P30R1, P32, P33, and P35 and must
report the P35-minus-native and P35-minus-P33 endpoint differences. It must
also verify that the full teacher target was retained and that no P34 target
shaping occurred.

## 8. One-attempt and no-rerun policy

There is exactly one possible future scientific Stage 2 attempt. No second
seed, rerun, restart after a result, learning-rate change, epoch change,
batch-size change, C change, weight-map change, target change, objective
change, threshold change, gate change, Stage 3, subset, or full run is
allowed. If a required gate fails, the future terminal state is
`P35_STAGE2_SCIENTIFIC_STOP`. Engineering failure is
`P35_STAGE2_ENGINEERING_STOP`. No automatic rescue is permitted.

No P35 scientific UUID, execution marker, held prediction, or scientific
result exists at this freeze.

## 9. Frozen implementation correspondence

The production module must expose one P35 objective contract with the exact
equation above. The reference module must use the same full deployment
algebra and must agree with production in FP32 within the preregistered
engineering tolerance. The implementation may reuse the P33 cached runner,
adapter, optimizer, schedule, and checkpoint structure, but it must not
fall back to P34 target shaping or alter P33/P34 evidence.

P35 changes only the source-example importance map. It does not change
inference, architecture, data, target, objective count, or optimizer.

P35 scientific Stage 2 attempts at freeze: `0`

P35 Stage 3 attempts at freeze: `0`

P35 full runs at freeze: `0`

P35 held tuning iterations at freeze: `0`

`P35_PREREGISTRATION_FROZEN`
