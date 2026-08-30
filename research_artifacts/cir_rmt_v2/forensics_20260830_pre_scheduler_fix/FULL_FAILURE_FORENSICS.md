# CIR_DFG_RMT_V2 full failure forensics

Decision: `KEEP_PARENT_FIX_TRAINING`

## Executive finding

The optimization audit is conclusive: `CIR_SCHEDULER_BUG_CONFIRMED`. CIR constructs and serializes StepLR but the epoch loop never calls `scheduler.step()`. The five CIR epoch checkpoints therefore retain `last_epoch=0` and initial image/text LRs instead of the intended gamma=0.9 decay.

This is a major protocol confound. Current CIR-V2 training was not optimization-matched to Phase2B, so the present benchmark cannot cleanly isolate the RMT hypothesis; the conclusion must not be `RMT failed`.

The preserved 90.98/40.35 comparison is not a matched V2 parent result. The historical table is E10 with pixel_stride=4, while the V2 artifact family starts at E12 and the current exact evaluator uses pixel_stride=1. The historical model also used a different objective and AMP setting. Therefore the anchor gap is not, by itself, evidence that the V2 RMT transport caused the failure.

The new alpha=0 control covers 30 medical checkpoint-target cells. At alpha=0.5 minus alpha=0, the aggregate summary is: `{"alpha05_better_both": 8, "alpha05_better_pixel_ap": 11, "alpha05_better_pixel_auroc": 10, "alpha05_worse_both": 17, "mean_delta_pixel_ap": -0.00034197302232319193, "mean_delta_pixel_auroc": -0.00015730314557440932, "n": 30}`.
Mean paired pixel AUROC is alpha=0 `0.8860026172612373` versus alpha=0.5 `0.8858453141156628`; mean pixel AP is alpha=0 `0.3368328745914157` versus alpha=0.5 `0.33649090156909256`. Metric values are decimal scores; the CSV contains the complete cell-level deltas.
Observed alpha=0 medical-matrix evaluator time summed across cells: `7789.1` seconds (sequential cell time; excludes source and gradient audits).

Decision rationale: CIR_SCHEDULER_BUG_CONFIRMED: current CIR-V2 training is not optimization-matched to Phase2B because train_full.py never calls scheduler.step(). The present benchmark cannot cleanly isolate the RMT hypothesis; run one matched corrective retrain before attributing degradation to RMT.

## Proven hard facts

- The current V2 checkpoints are E12/E14/E16/E18/E20 only; E10 is absent even though the parent candidate list includes E10.
- The current medical evaluator uses the deployed CIR map and exact full-resolution pixel metrics; colon image metrics are undefined and represented as null. The historical table used stride 4 and legacy zero-valued colon image columns.
- All five V2 checkpoints carry CIR_DFG_RMT_V2 identity, current parent config hash, FP32 metadata, and alpha=0.5 V2 direction. The nested adapter states drift across epochs.
- The source-confirmation alpha curve is a bounded 120-image VisA sign confirmation from a fresh parent model; its source code does not load a trained V2 checkpoint.
- The run manifest is weak provenance: it records an empty history and a producing commit different from the checkpoint producing commit. Checkpoint nested metadata is the stronger source for epoch/step/identity.

## Evidence status

- Proven: CIR scheduler bug (`scheduler.step()` absent from the epoch loop and stale serialized scheduler state), protocol mismatch, E10 disappearance, alpha=0 full-matrix control, current alpha=0.5 artifact identity, peer invariant measurements, deployment/training map difference, and checkpoint parameter drift.
- Correlational: association between RMT effect and medical metric changes; association between peer contamination or stage/group signal quality and final failure.
- Unknown: the magnitude of the scheduler confound after a matched corrective rerun, a matched Phase2B parent checkpoint under the current canonical config, full causal separation of training objective versus representation drift, and any claim that the old 90.98/40.35 result is reproducible under the current exact evaluator.

## Completed VisA-to-Medical failure map

The compact `medical_results_prefx.csv` preserves all 30 target/epoch rows for
Brain, Liver, Retina, Colon_clinicDB, Colon_colonDB, and Colon_Kvasir. The
six-medical pixel macro at alpha=.5 is:

| epoch | pixel AUROC | pixel AP |
|---:|---:|---:|
| 12 | 89.0550 | 35.3933 |
| 14 | 89.5530 | 35.5607 |
| 16 | 87.9140 | 32.0348 |
| 18 | 89.4628 | 34.2885 |
| 20 | 86.9373 | 30.9680 |

The published ACD-CLIP N=3 reference is 91.55 AUROC / 43.03 AP. The
per-target evidence preserves the following failure map without treating it
as a clean RMT causal result: Liver is the relative success/control domain;
Retina has relatively strong AUROC but materially weaker AP; Brain has
reasonable AUROC but severe pixel-AP weakness; ClinicDB is moderately and
systematically under target; ColonDB is a strong domain failure with late
checkpoint degradation; and Kvasir is systematically under target despite
relatively high AP compared with some other medical targets. E14 is relatively
strong, E16 degrades, E18 partially recovers, and E20 degrades again. These
patterns are joint evidence from the buggy-trained trajectory and deployment
evaluation, not an attribution directly to RMT.

## Optimization, loss, and deployment coverage

The exact current objective is:

```text
loss = cls_loss + seg_loss + lambda_kg * kg_loss + lambda_k * k_loss
```

with effective `lambda_kg=0.001` and `lambda_k=0.0`. The deterministic E14
gradient audit records classification norm 0.21690, segmentation norm 0.26503,
classification/segmentation cosine -0.01733, weighted-KG norm 0.001683, and
zero weighted-K norm. Clipping is once per optimizer step after accumulation.
All three named Adam groups use betas `(0.9, 0.999)`, eps `1e-8`, and zero
weight decay; the soft prompt uses a separate freeze/unfreeze constant-LR
policy. Resume code restores optimizer, scheduler, and RNG state, but a stale
serialized scheduler preserves the CIR bug rather than repairing it.

The train/deploy audit proves a path difference: the training map has no
deployment Gaussian blur, with mean absolute difference 0.004018, maximum
0.998923, and Pearson correlation 0.632972 on the deterministic E14 batch.
Its full-medical causal contribution is unknown. The peer audit preserves
K=8, spatial radius 3, peer distances, MAD, tanh/delta, native/transported
weights, and validity/finiteness fields; any GT-derived purity or contamination
is post-hoc `DIAGNOSTIC ONLY`. Stage/group attribution is preserved in 315
rows. No MVTec failure result was produced in this run; MVTec coverage is
`NOT_RUN`, not inferred.

## Coverage boundary

Completed coverage is scheduler/LR mismatch, paired alpha=0 versus alpha=.5,

## Root-cause ranking

| rank | candidate cause | evidence status | limiting evidence |
|---:|---|---|---|
| 1 | CIR trained without advancing StepLR | Proven, major protocol bug | No `scheduler.step()` in CIR loop; E12-E20 states all have `last_epoch=0` and initial image/text LRs; corrected-run effect size is not yet measured |
| 2 | Historical 90.98/40.35 anchor is not protocol-equivalent | Proven, high-impact confound | E10 vs E12+, stride 4 vs 1, and legacy objective/AMP differ |
| 3 | No matched parent control / different training lineage | Proven gap; causal contribution unknown | Required parent checkpoint hash is absent and current V2 adapter states drift |
| 4 | RMT transport effect in the current trained representation | Measured by the paired alpha matrix | Establishes an alpha contrast, not the cause of the absolute historical gap |
| 5 | Train/deploy map path divergence | Proven implementation difference; causal size unknown | Training path omits the deployment Gaussian blur |
| 6 | Peer signal quality or contamination | Measured post-hoc; correlational | Ground truth never enters peer selection; contamination is diagnosis only |
| 7 | Checkpoint/run-manifest provenance weakness | Proven administrative risk | Nested checkpoint identity is stronger than the empty run history |

The ranking separates demonstrated mismatches from hypotheses whose effect is only isolated by correlation or by the paired alpha intervention.

## Required output coverage

- `inference_rmt_effect.csv`: 30 medical cells, alpha=0 recomputed and alpha=0.5 read from identity-checked preserved artifacts.
- `checkpoint_drift.csv`: 5 checkpoints with nested adapter-state drift versus E12 and previous checkpoint.
- `peer_forensics.csv`: 35 full-target peer/mechanism summaries; GT contamination is post-hoc only.
- `stage_group_attribution.csv`: 315 bounded attribution rows over deterministic diagnostic images.
- `pixel_rank_forensics.csv`: 2100 deterministic representative image rows, with alpha=0 versus alpha=0.5 score changes and pixel metrics where defined.
- `gradient_conflict_report.csv`: 11 rows from a deterministic source training batch at E14.
- `scheduler_optimization_audit.csv` and `SCHEDULER_OPTIMIZATION_AUDIT.md`: serialized optimizer/scheduler audit classified `CIR_SCHEDULER_BUG_CONFIRMED`.

## Causal table

| hypothesis/intervention | evidence for | evidence against or limit | status |
|---|---|---|---|
| CIR trained with wrong LR schedule | CIR source omits `scheduler.step()`; serialized E12-E20 scheduler states have `last_epoch=0` and image/text LRs remain 1e-3/5e-4 instead of decaying | No corrected matched retrain yet; LR exposure is monotonic while the E14/E16/E18/E20 metric pattern is non-monotonic, so the bug's score contribution is unknown | Major proven confound; not an RMT inference effect |
| Historical anchor caused the observed V2 gap | Anchor uses E10/stride 4/legacy objective; V2 uses E12+/stride 1/current objective | No protocol-equivalent reproduction of the anchor | Not identified; confounded |
| Alpha=0.5 RMT transport changes the trained V2 result | Paired alpha=0 versus alpha=0.5 cells use the same checkpoint and evaluator | This isolates transport effect only within the current representation | Causal for the alpha contrast |
| Parent objective or representation drift caused the absolute gap | Config/objective/AMP and checkpoint lineage differ; adapter states drift | No matched parent checkpoint under the canonical config | Proven mismatch, causal size unknown |
| Deployment smoothing caused the final failure | Train and deploy maps are computed by different documented paths | One-batch audit does not isolate full medical impact | Proven path difference, causal size unknown |
| Peer quality or GT contamination caused the failure | Full-target peer statistics and post-hoc contamination are measured | GT is excluded from peer selection; association is not intervention | Correlational only |
| Epoch/checkpoint drift caused the failure | Nested adapter states and metrics change across E12-E20 | Epoch is confounded with optimization progress | Descriptive, not causal |

## Smallest next experiment and expected cost

Run one matched corrective parent/CIR training comparison under the current canonical config: same seed, VisA source, CLIP asset, FP32 policy, effective batch, Adam hyperparameters, StepLR gamma/step timing, losses, and E10/E12-E20 checkpoint schedule, with only CIR/RMT as the intended difference. Do not modify the frozen RMT architecture or start MVTec training.

Planning estimate: one matched 20-epoch VisA parent run on the RTX 5060 Ti 16GB is approximately 5-10 GPU-hours; this is an explicit planning range, not a measured rerun. The current alpha=0 medical matrix measured `7789.1` sequential evaluator seconds; a paired matrix should budget that amount for each newly evaluated alpha condition, subject to cache and loader variance.

## Limitation that controls the next experiment

Do not redesign the RMT architecture or launch MVTec training from this report. The smallest scientifically decisive next step is the matched corrective parent/CIR retrain above, followed by the same exact evaluator and a paired alpha=0 versus alpha=0.5 matrix. Until then, do not attribute the observed degradation directly to RMT.

All new files are isolated under this audit directory; existing V2 checkpoints and evaluation artifacts were not overwritten.
