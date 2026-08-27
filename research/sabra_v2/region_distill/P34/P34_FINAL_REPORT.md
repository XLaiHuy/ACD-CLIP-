# P34 Scientific Stage 2 Final Report

Final status: `P34_STAGE2_ENGINEERING_STOP`.

Exactly one preregistered candle attempt was created and completed the locked
39,240-step fit, prediction freeze, and post-freeze scoring path. The runner
then raised a reporting-wrapper `KeyError: 'source_only'` while assembling
the final Markdown report. The partial evidence is preserved; no second
attempt, repair-and-rerun, tuning, Stage 3, subset expansion, or full run is
permitted. The frozen scientific gate evidence before that wrapper failure was
`P34_STAGE2_SCIENTIFIC_STOP` because both detection endpoints failed their
preregistered thresholds.

## Frozen execution

- attempt UUID: `9a8d25f9-7592-479b-8fc8-2861226d7485`
- branch: `research/p29r1-fast-objective-forensic-v1`
- scientific execution commit: `5861badcef71118039b07e7bb5e2727c143d93be`
- engineering qualification commit: `81bb9ef3896bc4723bda9488b0a63a7a93cf2e33`
- preregistration SHA-256: `b78f69487e665b62d9c81b58da45f8f0afe5d047e91996f18569c6d38f99abdb`
- class/split: candle LOCO; fit `1962`; held `200`
- optimizer: AdamW, seed `0`, FP32, 20 epochs, batch size 1
- optimizer steps: `39240 / 39240`
- attempt span through wrapper stop: approximately `1196.833885 s`

## Detection

| method | pAP | pAUROC | pAP minus native | pAUROC minus native |
|---|---:|---:|---:|---:|
| P31/native | 0.514140304931 | 0.980667143514 | — | — |
| P30R1 | 0.511513734224 | 0.980534708954 | -0.002626570707 | -0.000132434559 |
| P32 | 0.510351502947 | 0.971460700418 | -0.003788801984 | -0.009206443096 |
| P33 | 0.519395095936 | 0.978184288830 | +0.005254791005 | -0.002482854684 |
| P34 | 0.505450138378 | 0.970397710904 | -0.008690166554 | -0.010269432610 |

P34 versus P33 was `-0.013944957558` pAP and `-0.007786577927` pAUROC.
The pAP and pAUROC gates therefore failed. Residual q99 and normal-score
shift gates passed: residual absolute q99 `4.913081169128` versus maximum
`8.643353872299`, and normal-score q99 shift `1.629600774322e-06` versus
maximum `0.001001158785112`.

## Selectivity and actionability

The same frozen mechanism epsilon, `C/100 = 0.0496010971069336`, was used for
the residual support comparison.

| method | active fraction | effective support fraction | Gini | top-10% mass |
|---|---:|---:|---:|---:|
| P30R1 | 0.111358024691 | 0.056409297806 | 0.925128442076 | 0.961142456400 |
| P32 | 0.871481481481 | 0.525447156883 | 0.505108044575 | 0.214533648140 |
| P33 | 0.999074074074 | 0.962760408648 | 0.069176234345 | 0.112053243823 |
| P34 | 0.875987654321 | 0.495237535130 | 0.523964988072 | 0.218834549686 |

P34 residual exact-nonzero fraction was `1.0`, but meaningful support fell
below P33, effective support fell below P33, and Gini rose above P33; all three
preregistered mechanism checks passed. The native-to-P34 score-effect q99
absolute value was `3.070960246987e-06`. Post-freeze descriptive anomaly
enrichment of active score effect was `303.059560547`, and is not a gate.

Frozen source-only actionability statistics were: weight exact-zero fraction
`0.208134238141`, weight-one fraction `0.438716992014`, weight `>0.75`
fraction `0.468221501837`, and weight `>0.9` fraction `0.452384477871`.
The shaped target exact-zero fraction was `0.208134238141`, near-zero fraction
`0.260839689814`, meaningful support `0.601229864420`, and the frozen sampled
target Gini was `0.514036400553`. Thus the explicit target reduced meaningful
intervention density relative to P33, while the endpoint result did not show
useful action preservation.

## Mechanism result

1. The P34 implementation used the explicit target `T=stop_gradient(w*E_t)`;
   algebraically, `w=0` supplies a zero target and a nonzero student effect
   receives a restoring gradient toward zero.
2. P34 was materially less dense than P33 under all three frozen mechanism
   checks.
3. Source-side actionable targets remained nonzero and included a frozen
   `w=1` population; however, P34 pAP/pAUROC did not preserve detection quality
   on the held candle fold.
4. The endpoint result does not support an all-zero/native success claim;
   P34 was worse than native on both endpoints.
5. Radial/tail and normal-score safety passed.
6. No raw teacher-vector fidelity objective or inference-time teacher was used.

## Audit and counts

- prediction freeze: `PASS`, UTC `2026-08-27T07:11:49.414712+00:00`
- prediction SHA-256: `6924601f55fb99dc98036667c2e5396510577ece5a4f63be8a6e3bc200df7497`
- held GT/mask reads before freeze: `0 / 0`
- held GT/mask reads after freeze: `200 / 100`
- student parameter delta L2: `60.26966676073134`
- frozen/teacher parameter delta: `0.0`
- loss/gradient finite: `true / true`; nonfinite counts: `0 / 0`
- new CLIP forwards: `0`
- new Phase2B forwards: `0`
- teacher forwards: `0`
- cache rebuilds: `0`
- reruns: `0`
- Stage 3: `false`
- full run: `false`
- post-run audit: `PASS`
- objective count: `1`
- new tuned hyperparameters: `0`

The engineering-stop cause and traceback are preserved in
`P34_STAGE2_ENGINEERING_STOP.json`; the qualification, prediction, freeze,
metrics, and downstream diagnostic artifacts are preserved without altering
the frozen preregistration or scientific implementation.

`P34_STAGE2_ENGINEERING_STOP`
