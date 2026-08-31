# Root-cause lock: pre-full-run CIR-V2 corrective decision

Status: LOCKED for the bounded pre-full-run decision, not a claim of final target-domain causality.

## Decision

Primary root cause: `R4_PIXEL_STAGE_REPRESENTATION_DRIFT`.

Secondary risk: `R10_DEPLOYMENT_MISMATCH`.

Selected solution for bounded implementation testing: `SELECTIVE_PHASE2B_ANCHOR`.

This lock is based on the corrected, scheduler-matched P/C0 baseline and the
same-checkpoint alpha comparison. It does not authorize a 20-epoch run, Medical
evaluation, MVTec evaluation, or target-driven tuning.

## Evidence that supports the primary lock

- The corrected P/C0 parameter audit shows the largest image-side movement in
  the image adapter (normalized L2 0.726 at E10 to 0.766 at E20), while text
  adapter movement is small (0.077 to 0.090). Soft-prompt movement is large,
  but its isolated E14 intervention is small.
- On the fixed 96-image VisA source sample, pooled segmentation cosine is
  0.894/0.878/0.807 at E10 for stages 0/1/2 and remains approximately
  0.897/0.888/0.767 at E20. Detector cosine is also low, especially at stage
  0 early in the sweep. The diagnostic classifies the feature-path drift as
  HIGH; these are same-image representation comparisons, not target claims.
- At E14, swapping only the CIR image adapter into the parent path changes
  source Pixel AUROC from 0.961153 (`PPP`) to 0.936674 (`CPP`). Swapping only
  text or only prompt leaves Pixel AUROC near 0.960. This is the strongest
  bounded intervention signal and localizes the dominant change to the image
  side of the pixel path.

## Evidence that supports the secondary risk

The repository's deployed map applies Gaussian smoothing and differs from the
raw training probability. At E18 the C raw-to-deployed source AP drop is
0.167088 versus 0.068816 for P. This makes deployment mismatch a meaningful
secondary risk, but the effect is shared by P and C and is not consistently
C-specific across epochs. The fraction of the target failure explained by this
operator is therefore unknown.

## Findings explicitly not locked as primary causes

- `R1` overspecialization is not proven. The category-heldout gate is mixed,
  and the training set contained all categories, so this is an assessment split
  rather than a true unseen-category experiment.
- `R2` and `R3` are not primary by the E14 module swap. Text-only and prompt-only
  changes are materially smaller than the image-only change.
- `R5` compensation is not supported. On the current representations, alpha
  0.5 changes probabilities only at numerical-small scale; this is a negative
  inference signal, not proof that every future RMT variant is intrinsically
  weak.
- `R12` generic optimization failure is not supported after the scheduler
  correction. The corrective P and C0 runs are optimizer/scheduler matched.

## What is proven, correlational, and unknown

Proven within the bounded measurement scope:

1. The pre-fix CIR scheduler bug was real and the corrective P/C0 training
   protocol is matched.
2. C0 and P have materially different image-side learned parameters and
   same-image segmentation/detection representations.
3. The image-adapter-only intervention is the dominant E14 source-path change.
4. Current alpha=.5 inference is neutral relative to alpha=0 on the paired
   source measurements and the preserved full Medical pairs.
5. Gaussian deployment changes the reported pixel map and can cause a large AP
   loss, including a C-specific E18 episode.

Correlational or intervention-supported but not target-causal:

1. Image-side representation drift is associated with source metric differences
   and is the best current root-cause localization.
2. The E18 C high-normal-tail event is associated with a source AP collapse.
3. Deployment mismatch may contribute to late AP degradation.

Unknown:

1. How much of the Medical gap is caused by image representation drift versus
   target-domain robustness, loss balance, or deployment behavior.
2. Whether an image parameter anchor will improve target transfer without
   suppressing useful CIR learning.
3. Whether a redesigned/non-neutral RMT can provide a useful inference signal.

## Scientific interpretation

The scheduler bug invalidated the old direct comparison as a clean RMT test.
After scheduler correction, the bounded evidence does not support attributing
the remaining C0-vs-P target degradation directly to RMT inference: C05-C0 is
neutral. The current lock instead identifies the learned image/pixel
representation path as the most actionable measured mechanism, with deployment
as a secondary risk.
