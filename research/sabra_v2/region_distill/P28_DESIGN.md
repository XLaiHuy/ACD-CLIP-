# P28 — P27 Region Distillation Mechanism Diagnostic V1

P28 is a post-hoc, zero-training diagnostic of the completed and immutable
P27 attempt. It separates the observed P27 effect into four frozen states:

`NATIVE (N) -> PATCH-TEACHER ORACLE (OP) -> REGION-TEACHER ORACLE (OR) -> STUDENT (S)`.

N and S come directly from the immutable P27 prediction artifacts. OP uses
the held VisA mask only after the P27 freeze, the frozen Tier-A native logits,
and the exact historical R0 signed utility with alpha 0.25 and margin scale
19.840438842773438. OP is diagnostic/oracle evidence and is not a predictor.
OR applies the exact P27 adaptive average pooling, aligned-bilinear
reconstruction, and symmetric two-class residual to OP. S is never
regenerated: its prediction maps are loaded from the immutable artifacts.
The P27 adapter is run only on already-frozen Tier-A segmentation features to
recover its 9x9 output for teacher/student alignment; this is inference, not
training or a new CLIP/Phase2B forward.

The primary causal contrasts are OP-N (teacher semantics), OR-OP
(regionization), S-OR (student transfer), and S-N (the observed P27 effect).
Each state is scored per class and macro with the exact P27 rank-group pAP and
pAUROC definitions. Ranking diagnostics quantify anomaly and normal score
shifts, gained/lost anomaly-over-normal orderings, and fixed top-rank anomaly
fractions. Residual diagnostics split anomaly and normal pixels and report
normal upper-tail and anomaly lower-tail behavior. Alignment compares the
held OR target with the frozen P27 adapter output at 9x9 using Pearson,
Spearman, sign agreement, MAE, and a preregistered robust magnitude ratio,
without fitting or calibration.

The five preregistered hypotheses concern teacher objective conflict,
regionization ranking loss, student transfer failure, normal-score inflation,
and heterogeneous category actionability. The primary root-cause label is
chosen only after inspecting the full 12-class decomposition and is one of
the preregistered labels in `P28_PROTOCOL.json`. All correlations across the
12 categories are exploratory. P28 does not select or promote a deployable
model and does not implement P29.

Inputs are restricted to the frozen P27 artifacts, Tier-A cache, P27 adapter
checkpoints, VisA metadata, and post-hoc held VisA masks. MVTec and Medical
are firewalled. There is exactly one diagnostic execution, with no training,
optimizer steps, parameter updates, new CLIP forwards, new Phase2B forwards,
parameter sweeps, thresholds, checkpoint selection, or result-driven rerun.
