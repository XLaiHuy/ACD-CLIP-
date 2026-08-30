# CIR-V2 research direction — pre-fix forensic snapshot

This is a pre-corrective-training interpretation of the completed CIR-V2
forensics. It preserves the scheduler mismatch as a first-order confound and
does not propose an architecture change.

## 1. WHAT FAILED

The current CIR-V2 benchmark did not deliver a protocol-equivalent transfer
comparison against Phase2B. The CIR epoch checkpoints show unstable medical
behavior: E14 is relatively strong, E16 drops, E18 partially recovers, and
E20 drops again. The six-medical pixel macro is:

| epoch | pixel AUROC | pixel AP |
|---:|---:|---:|
| 12 | 89.0550 | 35.3933 |
| 14 | 89.5530 | 35.5607 |
| 16 | 87.9140 | 32.0348 |
| 18 | 89.4628 | 34.2885 |
| 20 | 86.9373 | 30.9680 |

The published ACD-CLIP N=3 reference is 91.55 AUROC / 43.03 AP, but the
historical comparison is not a matched current V2 parent control.

## 2. WHAT WAS INVALIDATED BY THE SCHEDULER BUG

The CIR loop constructs `StepLR(step_size=1, gamma=0.9)` but never calls
`scheduler.step()` in its epoch loop. E12/E14/E16/E18/E20 retain
`last_epoch=0`, `_step_count=1`, image LR `1e-3`, and text LR `5e-4`.
Relative to the intended parent start-of-epoch schedule, the image/text
exposure is approximately 3.19x, 3.93x, 4.86x, 6.00x, and 7.40x at those
epochs. Therefore the absolute CIR-versus-Phase2B gap, late-epoch checkpoint
drift, source over-specialization, and transfer degradation are not clean
RMT effects. The current benchmark cannot support the statement “RMT failed.”

## 3. WHAT STILL APPEARS TO BE A REAL RMT-SPECIFIC SIGNAL

The paired alpha intervention is real as an inference contrast: for each
same-checkpoint cell, `V05 - V0` measures the alpha=.5 transport effect
conditional on the wrongly trained representation. Across 30 medical cells,
alpha=.5 is better on both pixel AUROC and AP in 8 cells and worse on both in
17; mean deltas are -0.0001573 AUROC and -0.0003420 AP. This is a small,
mixed signal, not a clean CIR-vs-Phase2B causal estimate. The source-side and
mechanism tables preserve the RMT transport behavior for rechecking after
matched training.

## 4. WHAT CANNOT YET BE ATTRIBUTED TO RMT

The historical anchor gap, the E12-to-E20 representation trajectory, the
Brain/colon transfer failures, and the non-monotonic checkpoint pattern are
all confounded by excessive CIR LR exposure and other protocol/lineage
differences. The available parent checkpoint is legacy and has no serialized
optimizer or scheduler state. The alpha matrix does not identify how a
corrected training trajectory would behave.

## 5. WHAT THE LOSS/DEPLOYMENT/PEER AUDITS SUGGEST

The audited objective is exactly:

```text
loss = cls_loss + seg_loss + lambda_kg * kg_loss + lambda_k * k_loss
```

The current effective values are `lambda_kg=0.001` and `lambda_k=0.0`.
The deterministic E14 batch records classification gradient norm 0.21690,
segmentation gradient norm 0.26503, and classification/segmentation gradient
cosine -0.01733; weighted KG is small and weighted K is zero. Gradient
clipping is once per optimizer step after accumulation. All three named Adam
groups use betas (0.9, 0.999), eps 1e-8, and zero weight decay; the soft prompt
has a separate freeze/unfreeze constant-LR policy.

The train/deploy map audit proves a path difference: mean absolute difference
0.004018, maximum 0.998923, and Pearson correlation 0.632972 on the
deterministic E14 batch. Training omits the deployment Gaussian blur. Its
full-medical causal contribution remains unknown.

The RMT mechanism uses K=8 peers, spatial radius 3, midpoint-median center,
MAD scale, tanh transform, and detached peer deltas. Peer validity, spatial
constraints, and finiteness passed in the preserved rows; MAD and delta
distributions, transport/native weights, stage, and group outputs are
recorded. Large delta/z values and near-zero MAD cases are diagnostic signals
for saturation sensitivity, not proof of causal failure. Any post-hoc GT
purity/contamination field is DIAGNOSTIC ONLY and cannot become a tuning rule.

## 6. WHAT MUST BE RECHECKED AFTER CORRECTIVE RETRAIN

Repeat the exact evaluator and compare parent/CIR at the same epochs with the
correct StepLR timing. Recheck serialized optimizer/scheduler states, source
and six-medical metrics, E12-to-E20 drift, paired alpha=0 versus alpha=.5,
train/deploy maps, gradient conflict, stage/group attribution, peer purity,
MAD, delta magnitude, saturation, and checkpoint selection. MVTec remains
outside this pre-fix evidence set and must not be inferred from it.

## 7. WHAT WOULD JUSTIFY KEEPING RMT

A matched corrected parent/CIR comparison in which the CIR representation is
competitive and alpha=.5 is neutral or beneficial on held-out target metrics,
with stable peer diagnostics and no unacceptable deployment-path artifact,
would justify keeping the frozen RMT mechanism for further study.

## 8. WHAT WOULD JUSTIFY REDESIGNING RMT

If the corrected protocol still shows repeatable harm, the next question is
mechanism design: peer quality, MAD near-zero handling, tanh saturation,
transport magnitude, stage/group concentration, and the train/deploy operator
should be tested with bounded diagnostics. Redesign would require evidence
from the corrected run, not the current scheduler-confounded trajectory.

## 9. WHAT WOULD JUSTIFY ABANDONING RMT

Abandonment would require a protocol-matched, reproducible alpha=.5 harm (or
no useful signal) across the relevant targets and checkpoints after the
optimization mismatch is repaired, with the parent control and deployment
operator held fixed. The current snapshot does not meet that standard.

## 10. THE SINGLE NEXT EXPERIMENT

Run one matched corrective Phase2B-vs-CIR retrain with identical:

- seed
- VisA source
- CLIP asset
- FP32
- effective batch
- Adam config
- LR
- StepLR
- `scheduler.step()` timing
- loss
- prompt schedule
- DFG schedule
- checkpoint schedule

The only intended difference is CIR/RMT. Do not launch that experiment as
part of this snapshot task.

Current decision: `KEEP_PARENT_FIX_TRAINING`.
