# P31 vs P30R1 Causal Analysis

Status: `P31_CONTROL_ALREADY_EXISTED`

This report completes the P31 native/zero-adapter control analysis without a
new execution. The exact native counterfactual and paired candle metrics were
already produced by the immutable P30R1 scoring/forensic path. Repeating that
held comparison would duplicate an answered control question.

## 1. Entry and provenance

- Branch: `research/p29r1-fast-objective-forensic-v1`
- Analysis HEAD: `905f325232ccdb83d20b6c7cb30e7300eae3ef47`
- P31 engineering commit: `1f6d1d68c04ce4fde49cec86bbe5d3b9a22cc85d`
- P31 preregistration SHA-256: `f42f0add36c0de2e303e6f25b0d48b63c33eda7d4c56d2a7ccb368ca76c865e3`
- Parent forensic status: `FORENSIC_COMPLETE`
- Held class: `candle`; records: `200`
- Existing frozen prediction source: `P30R1/candle/predictions/p30r1_held_predictions.pt`
- Existing source prediction SHA-256: `30c250b52ff980e7b16fa0e97ffaebb19e13fd5521722e5397cc86ed2c4e1218`

The source prediction artifact is `PREDICTIONS_FROZEN_GT_FREE`; its native
probability field is the exact P31 identity output. The forensic also
reconstructed the same native output from frozen Tier-A logits with maximum
absolute map error `1.1920928955078125e-07`, below the inherited `2e-5`
engineering tolerance. The field-level freeze record is
[`P31_PREDICTION_FREEZE.json`](P31_PREDICTION_FREEZE.json).

The frozen P31 preregistration already defines the identity control, the
native-vs-P30R1 endpoints, the zero-margin rule, and the no-rerun rule. No
separate control preregistration was needed because this report does not open a
new held scoring pass.

## 2. Primary comparison

The stored native reference values are retained alongside the independently
reconstructed native cross-check:

| Metric | Native/P31 | P29 | P30 | P30R1 |
|---|---:|---:|---:|---:|
| pAP | 0.514140305 | 0.490503231 | 0.144618064 | 0.511513734 |
| pAUROC | 0.980667144 | 0.970040297 | 0.972904419 | 0.980534709 |
| Mean abs residual | 0 | 2.036683 | 1.542490 | 0.178744 |
| Residual q99 | 0 | 4.321677 | 25.929799 | 4.528307 |
| Normal score-delta q99 | 0 | 1.16e-6 | 0.998691 | 7.33e-6 |

Using the stored native reference, the locked effect of P30R1 relative to
P31/native is:

```text
Delta pAP    = pAP(P31/native) - pAP(P30R1)    = +0.002626570707
Delta pAUROC = pAUROC(P31/native) - pAUROC(P30R1) = +0.000132434559
```

Both differences are nonnegative under the frozen zero-margin rule. The
reconstructed native cross-check gives `+0.002626567385` pAP and
`+0.000132435263` pAUROC, with the same conclusion.

P30R1 remains `STAGE2_SCIENTIFIC_STOP`; this control does not retroactively
change that status into a pass.

## 3. Prediction similarity and ranking

These are descriptive comparisons of the already-frozen native and P30R1
maps, not independent-pixel significance tests and not tuning criteria.

| Descriptor, P30R1 vs native | Value |
|---|---:|
| Pooled Pearson | 0.731909455 |
| Pooled Spearman, average ties | 0.974893733 |
| Mean per-image Pearson | 0.947845739 |
| Mean per-image Spearman | 0.907219412 |
| Mean absolute score difference | 0.001114394 |
| Median absolute score difference | 3.12e-11 |
| q90 absolute score difference | 4.20e-9 |
| q95 absolute score difference | 3.68e-8 |
| q99 absolute score difference | 1.37e-5 |
| Maximum absolute score difference | 0.935700119 |
| Top-0.1% overlap | 0.946115 |
| Top-0.5% overlap | 0.963461 |
| Top-1% overlap | 0.956680 |
| Top-5% overlap | 0.869183 |

The high pooled rank agreement and top-1% overlap show that P30R1 is mostly
native-like in ordering, but the tail is not identically preserved. The
native/P30R1 q99 score values are `5.81e-7` and `1.43e-5`; at q99.9 they are
`0.0546` and `0.8465`. Exact all-pairs ordering enumeration was not performed
because it is O(N²); pooled/per-image rank correlations and fixed top-quantile
overlaps are the preregistered descriptive evidence.

## 4. Correction effect and selectivity

P31 has no correction by construction:

```text
residual                 = 0 exactly
score delta              = 0 exactly
effective support        = 0
inference overhead       = 0%
```

The frozen P30R1 intervention is small for most pixels but not globally
zero:

- mean absolute score delta: `0.001114394`;
- median absolute score delta: `3.12e-11`;
- q99 absolute score delta: `1.37e-5`;
- maximum absolute score delta: `0.935700119`;
- `98.257%` of pixels have absolute delta at most `1e-6`;
- residual effective support fraction: `0.056409`;
- residual top-1%, top-5%, and top-10% absolute-mass fractions:
  `0.303896`, `0.823552`, and `0.961142`;
- positive/negative/exact-zero score-delta fractions:
  `0.917090` / `0.082877` / `0.0000336`.

The prior post-freeze mask analysis found mean absolute score delta
`0.000675487` on normal pixels and `0.325263709` on anomaly pixels, with
`291.8757x` anomaly-area enrichment. This supports descriptive spatial
selectivity, not a claim that the correction adds causal value: native is
still slightly better on both locked downstream metrics.

## 5. Direction metrics are not a P31 criterion

P31 has no student residual or teacher target, so its teacher-direction
cosine, sign agreement, and residual Spearman are
`NOT_MEANINGFUL_FOR_ZERO_ADAPTER_CONTROL`.

The historical contrast remains informative:

| Method | Directional cosine | Sign agreement | Residual Spearman | pAP |
|---|---:|---:|---:|---:|
| P30 | 0.736924 | 0.569568 | 0.714234 | 0.144618 |
| P30R1 | -0.070148 | 0.119691 | 0.054565 | 0.511514 |
| P31 identity | N/A | N/A | N/A | 0.514140 |

P30 has better raw teacher-direction metrics but much worse detection. P30R1
recovers detection while its raw direction metrics collapse. Raw teacher
direction is therefore not a validated downstream proxy.

## 6. Causal interpretation

The P31/native control is non-inferior to P30R1 on the locked candle comparison.
The strongest scoped interpretation is that P30R1 did not demonstrate a
downstream benefit beyond the native detector; its small, anomaly-enriched
changes are intervention evidence, not proof of useful correction. P31 also
preserves normal-score behavior exactly, while P30R1 only approximately
recovers it from P30's tail failure.

The P31 control does not establish a universal claim across categories. No new
12-category predictions were generated. Historical learned transfers improved
pAP in some categories while regressing AUROC broadly, so no cross-category
safety claim is made.

**Primary causal mechanism:** `TEACHER_DIRECTION_NOT_CAUSAL`

**Secondary mechanism:** `SPARSE_SELECTIVE_CORRECTION`

**Next research question:** What downstream-relevant invariant should be
transferred instead of raw teacher correction direction?

**Decision:** `KEEP_CURRENT` — keep the native detector as the default for this
locked scope and stop teacher-imitation expansion. Any future learned
downstream-functional transfer requires a new research decision and
preregistration; do not add a rescue direction, sign, ranking, gating, or
feature-consistency loss from this control alone.

## 7. Execution and audit counts

This turn performed no scientific execution. The prior P30R1 held mask reads
(`100`) occurred after its prediction freeze and belong to the comparator, not
to P31.

```text
P31 training runs                         = 0
P31 optimizer steps                      = 0
new CLIP forwards                        = 0
new Phase2B forwards                     = 0
new teacher forwards                     = 0
P31 adapter forwards                     = 0
cache rebuilds                           = 0
P31 held GT reads this turn              = 0
P31 held mask reads this turn            = 0
new scientific scoring passes            = 0
reruns                                   = 0
scientific UUIDs                         = 0
execution/scientific markers             = 0
Stage 3 attempts                         = 0
full/12-class runs                       = 0
held-result tuning iterations            = 0
MVTec touched                            = false
Medical touched                          = false
```

The machine-readable result, detailed metrics, freeze manifest, and post-run
audit are:

- [`P31_CONTROL_SCIENTIFIC_RESULT.json`](P31_CONTROL_SCIENTIFIC_RESULT.json)
- [`P31_VS_P30R1_CAUSAL_ANALYSIS.json`](P31_VS_P30R1_CAUSAL_ANALYSIS.json)
- [`P31_PREDICTION_FREEZE.json`](P31_PREDICTION_FREEZE.json)
- [`P31_CONTROL_POST_RUN_AUDIT.json`](P31_CONTROL_POST_RUN_AUDIT.json)

Required terminal state: `P31_CONTROL_ALREADY_EXISTED`

`NO TRAINING WAS PERFORMED.`

`NO STAGE 3 OR FULL RUN WAS STARTED.`
