# Post-diagnostic prior-art screen

This screen was written after the root-cause lock. It is a design screen, not
evidence that any method will improve this project. No target benchmark was
used to choose a candidate.

## Methods/families considered

1. **L2-SP / selective parameter anchoring.** Li et al., “Explicit Inductive
   Bias for Transfer Learning with Convolutional Networks,” PMLR 80 (2018),
   https://proceedings.mlr.press/v80/li18a.html. This is the closest fit to the
   measured image-side parameter and feature drift. A selective image-only
   anchor preserves the existing Adam groups, StepLR, FP32 policy, RMT forward
   path, and deployment operator.
2. **Feature-level consistency regularization.** Pseudo-label/feature
   consistency families such as PseudoSeg (Zhang et al., 2021),
   https://openreview.net/forum?id=y8hxlSQg03b. A feature anchor is mechanistic
   for R4 but would require a stable reference representation and adds a second
   forward/reference path, so it is not selected for this bounded implementation.
3. **Deployment-consistent training.** Training against the exact deployed
   Gaussian/resize operator is a plausible response to K7, but the operator is
   shared by P and C and the C-specific effect is not consistent across epochs.
   It remains a follow-up only if the selected anchor fails and K7 becomes the
   dominant red-team result.
4. **RMT/SAR redesign or stronger transport.** The current paired alpha effect
   is numerically neutral and peer deltas are highly saturated. A transport
   redesign is therefore not justified before correcting the image path and
   re-establishing a non-neutral signal.

## Constraints carried forward

The candidate must preserve Adam hyperparameters, three optimizer groups,
StepLR gamma/timing, FP32, effective batch six, seed, VisA source, CLIP asset,
loss terms, prompt schedule, DFG schedule, checkpoint schedule, evaluator, and
Gaussian deployment. A candidate that changes more than the selected cause is
not a clean next experiment.
