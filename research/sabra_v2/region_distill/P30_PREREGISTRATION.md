# P30 Speed–Performance Directional Distillation

## Hypothesis

P29R1 found supported mixed-objective conflict and secondary gradient
starvation. P30 tests the narrowest follow-up: preserve the signed spatial
correction direction while removing P29’s simultaneous magnitude, sign, and
normal auxiliary objectives.

## Exact candidate

The unchanged P29 `RegionResidualAdapter` emits `S = [3,B,9,9]`. The cached
teacher target is `teacher_region = [B,9,9]`, staged as
`T = teacher_region.unsqueeze(0).expand_as(S)`. For each sample, flatten the
normalized staged tensors `s = S/C` and `t = T/C` into 243 coordinates and use
the fixed `epsilon = 0.01`:

```text
r_s = sqrt(mean(s²) + epsilon²)
r_t = sqrt(mean(t²) + epsilon²)
s_hat = s / r_s
t_hat = t / r_t
L_P30 = mean_valid(1 - mean(s_hat * t_hat))
```

An exactly zero teacher vector is excluded because it has no defined direction.
No SmoothL1, sign, normal, ranking, feature, calibration, or other auxiliary
loss is present. The teacher is detached. The objective is training-only.

## Frozen experiment

P30 retains the P29/P27 adapter, cached Tier-A/Tier-B float32 inputs, historical
R0 teacher, 12 fixed LOCO folds, FP32 policy, AdamW (`lr=0.001`), 20 epochs,
batch size 1, seed 0, and unchanged P26 deployment. Only the adapter is
trainable. No MVTec or Medical data is permitted.

Stages are fixed before observing P30 outcomes: one-step `candle` smoke, full
one-class `candle` qualification, four-class subset
`candle/chewinggum/macaroni2/pcb3` (canonical-order positions 0/3/6/9), then
the full 12-class run. A separate UUID and marker identify the full P30
experiment.

## Gates and reporting

Before training, synthetic tests must cover identical/opposite direction,
magnitude scales `0.1×/1×/10×/100×`, zero and near-zero targets, partial sign
mismatch, ordering, and backpropagation. The student output gradient must be
finite, non-zero, and non-explosive. Smoke training must change the student,
leave the teacher unchanged, and save/reload a checkpoint.

The scorecard reports pAP, AUROC, sign and cosine direction, Pearson/Spearman
agreement, residual magnitudes, normal/anomaly shifts, category regressions,
gradient health, training/evaluation/inference time, memory, and objective
complexity. The preferred training overhead is at most 10%; inference overhead
is targeted at 0%. No tuning follows observed results.

Full acceptance is balanced: both primary metrics must be at least P29 for the
strong “better” decision, with improved direction and stable normal scores.
An equivalent-but-simpler or faster result is acceptable under the fixed
tolerances in the machine-readable preregistration. Otherwise preserve the
negative result and stop.
