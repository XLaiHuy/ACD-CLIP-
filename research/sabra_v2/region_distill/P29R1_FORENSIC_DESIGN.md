# P29R1 — Fast Objective / Transfer Forensic V1

P29R1 is a sealed zero-training diagnostic rooted at P29 terminal commit
`7eeee454538cb997496f8cd1107f66fa73a9c876`. It exists solely to explain why
P29's sign-guarded, normalized objective did not recover native AUROC and was
weaker than P27. It does not change, replay, rescore, or reinterpret P29.

## Recovery boundary

The historical P29 post-hoc audit failed because `utility_for_batch` was called
from a `torch.no_grad()` scope even though its R0 action computation uses
`torch.autograd.grad`. The forensic helper will place only that exact call
inside `torch.enable_grad()`. Native/cached tensors remain detached; no
optimizer is constructed and no model is updated. Historical R0 actions,
teacher targets, P29 checkpoints, P29 predictions, and P29 scores are frozen.

## Evidence path

The implementation will add a narrow P29R1 forensic module plus targeted tests.
It will use Tier-A cache (`seg_features`, `native_logits`, manifests), Tier-B
source targets/masks, P29/P27 frozen checkpoints and predictions, and retained
P28R1 OR evidence. It processes one held class at a time and source gradient
probes are bounded to the fixed four source classes and at most eight samples
per class. There are no new CLIP or Phase2B forwards.

The fixed analysis order is:

1. Verify the recovered R0 action call and artifact hashes.
2. Compare held 9x9 P29 teacher/student alignment and P27/P29 residual sizes.
3. Probe P29 value/sign/normal gradients at zero initialization and frozen
   P29 state, alongside raw P27 distillation and P27 segmentation gradients.
4. Measure source teacher-positive activity in exact P29 pure-normal regions.
5. Measure held normal/anomaly pixel shifts using only frozen deployed maps.
6. Reuse frozen score/OR references for descriptive recovery ratios.

## Test and performance gates

Tests must first demonstrate the old no-grad failure and then prove the local
enable-grad recovery; they also enforce no optimizer, CLIP, Phase2B, MVTec, or
Medical path, fixed hashes/inventory, fixtures for every statistic, and the
runtime estimator. A metadata/small-cache preflight measures throughput,
projected runtime, RAM and GPU use without scientific outputs. A projection of
at most 45 minutes permits execution. A 45–90 minute estimate permits exactly
one I/O/vectorization optimization and re-benchmark; a remaining estimate over
90 minutes stops before the forensic marker.

## Frozen execution and terminalization

After tests, pre-audit and qualified preflight, implementation and protocol are
committed/pushed as the P29R1 execution base. Exactly one marker records input
hashes and UUID immediately before all real diagnostic quantities are computed.
No post-marker patch or rerun is allowed. Small JSON/CSV/Markdown evidence is
then audited, committed and pushed. The sole terminal statuses are
`P29R1_FORENSIC_COMPLETE`, `P29R1_PERFORMANCE_STOP`, and
`P29R1_ENGINEERING_STOP`; P30 is out of scope.
