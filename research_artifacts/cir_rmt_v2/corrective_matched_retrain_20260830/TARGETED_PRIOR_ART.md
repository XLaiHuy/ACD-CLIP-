# Bounded targeted prior-art note

Status: COMPLETE AS A SCOPED REVIEW; no method was imported or implemented.

The review was deliberately limited to the locked K2/K3/K7 bottleneck. It is a design-context note, not evidence that any cited technique will improve this experiment.

## 1. Starting-point / representation preservation

Li, Grandvalet, and Davoine, “Explicit Inductive Bias for Transfer Learning with Convolutional Networks” (ICML 2018), proposes using the starting model as the reference for an L2-SP penalty rather than pulling parameters toward zero. That is the closest conceptual match to preserving Phase2B transfer geometry while adapting on VisA. It would still need a source-only, one-variable test; the paper does not establish the correct penalty for this CLIP/DFG protocol.

Source: https://proceedings.mlr.press/v80/li18a.html

## 2. Structured prediction consistency

PseudoSeg, “Designing Pseudo Labels for Semantic Segmentation” (ICLR 2021), and related consistency-regularization work use consistency across structured segmentation predictions and carefully calibrated pseudo-labels. This is a conceptual fit for testing whether the training-side map should remain consistent with a frozen deployment-side operator, but the setting differs: the current run is supervised VisA adaptation, not semi-supervised target adaptation. No pseudo-labeling or extra head is authorized.

Source: https://openreview.net/forum?id=y8hxlSQg03b

## 3. AP-sensitive anomaly evaluation

Rafiei, Breckon, and Iosifidis, “On Pixel-level Performance Assessment in Anomaly Detection,” highlights severe pixel imbalance and argues that precision-recall metrics can reveal localization weaknesses that AUROC can mask. This supports keeping Pixel AP as a primary gate and motivates the unrun AP-tail diagnostic; it is an evaluation argument, not a proposed fix.

Source: https://arxiv.org/abs/2310.16435

## Scope conclusion

The literature does not justify stacking a new RMT module before the current evidence is resolved. The leading future test is a minimal representation-preservation variable using the native Phase2B deployment path. A deployment-consistency variable is secondary. SAR-RMT remains unauthorized because the current C05−C0 effect is neutral and peer delta saturation is not proof of a useful latent signal.
