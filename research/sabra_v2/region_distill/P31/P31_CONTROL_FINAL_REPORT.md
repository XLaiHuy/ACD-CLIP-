# P31 Control Final Report — Native / Zero-Adapter vs P30R1

Final status: `P31_CONTROL_ALREADY_EXISTED`

## Result

The P31 zero-adapter control was already present as the exact native
counterfactual in the immutable P30R1 candle evidence. No independent P31
scientific run was opened and no held scoring was repeated.

| Metric | Native/P31 | P30R1 | Native − P30R1 |
|---|---:|---:|---:|
| pAP | 0.514140304931 | 0.511513734224 | +0.002626570707 |
| pAUROC | 0.980667143514 | 0.980534708954 | +0.000132434559 |

The native control is non-inferior on both frozen endpoints under the locked
zero-margin rule. The reconstructed native counterfactual independently gives
pAP `0.514140301610` and pAUROC `0.980667144217`, with maximum map error
`1.19e-7` against the stored native field, below the `2e-5` tolerance.

## What P30R1 added beyond native

P30R1 is highly rank-similar to native but not identical:

- pooled Pearson: `0.731909455`;
- pooled Spearman: `0.974893733`;
- top-1% overlap: `0.956680`;
- mean absolute score delta: `0.001114394`;
- delta q99: `1.37e-5`;
- maximum score delta: `0.935700119`.

Its residual effective support is `0.056409`, and prior post-freeze mask
analysis found `291.8757x` anomaly-area enrichment. Thus the correction is
small and spatially selective, but it is not shown to be useful beyond native:
native still wins both locked downstream metrics.

P31 has exact zero residual and exact zero score delta. Its teacher-direction,
sign, and residual-ranking metrics are not meaningful because it has no
student residual or teacher target. The historical P30/P30R1 contrast remains
the evidence against raw teacher direction as a downstream objective: P30 has
directional cosine `0.736924` but pAP `0.144618`, whereas P30R1 has cosine
`-0.070148` and pAP `0.511514`.

## Primary causal conclusion

`TEACHER_DIRECTION_NOT_CAUSAL`

Secondary: `SPARSE_SELECTIVE_CORRECTION`.

The exact control supports keeping the native detector as the default for this
locked scope. P30R1 remains `STAGE2_SCIENTIFIC_STOP`; P31 does not relabel it.
No feature-consistency term, ranking term, sign term, gate, or other rescue
objective is justified by this control.

Next research question: **What downstream-relevant invariant should be
transferred instead of raw teacher correction direction?** Any follow-up needs
its own research decision and preregistration.

## Data and execution audit

P31 used only existing frozen artifacts:

- frozen P30R1 prediction artifact, 200 candle records;
- its native probability field as the P31 identity output;
- P30R1 held metrics and P30R1 causal forensic report;
- no new images, labels, masks, model forward, or cache build.

```text
P31 training runs                  = 0
optimizer steps                   = 0
new CLIP forwards                 = 0
new Phase2B forwards              = 0
teacher forwards                  = 0
cache rebuilds                    = 0
held GT reads by P31 this turn    = 0
held mask reads by P31 this turn  = 0
new scientific scoring passes     = 0
reruns                            = 0
scientific UUIDs                  = 0
execution markers                 = 0
Stage 3 / full runs               = 0
```

The prior comparator read 100 held masks only after its frozen prediction
stage; this is historical P30R1 activity, not P31 activity.

Detailed evidence:

- [`P31_CONTROL_SCIENTIFIC_RESULT.json`](P31_CONTROL_SCIENTIFIC_RESULT.json)
- [`P31_VS_P30R1_CAUSAL_ANALYSIS.md`](P31_VS_P30R1_CAUSAL_ANALYSIS.md)
- [`P31_VS_P30R1_CAUSAL_ANALYSIS.json`](P31_VS_P30R1_CAUSAL_ANALYSIS.json)
- [`P31_PREDICTION_FREEZE.json`](P31_PREDICTION_FREEZE.json)
- [`P31_CONTROL_POST_RUN_AUDIT.json`](P31_CONTROL_POST_RUN_AUDIT.json)

`NO TRAINING WAS PERFORMED.`

`NO STAGE 3 OR FULL RUN WAS STARTED.`
