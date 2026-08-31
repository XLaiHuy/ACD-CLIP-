# Post-corrective diagnosis: what measurable bottleneck remains?

## Answer

The remaining measurable bottleneck is target-domain pixel transfer, especially Pixel AP. Corrected C0 is source-strong but Medical-pixel-weaker than P; C05 does not recover it. The current evidence supports a representation-preservation/generalization problem with an independently measured train/deploy operator risk. It does not support a meaningful positive RMT inference effect.

## Case A–K classification

| case | classification | evidence-based reading |
|---|---|---|
| A — clean RMT win | NOT_SUPPORTED | C0 is not close to P on Medical pixel metrics; C05−C0 is not positive/material. |
| B — RMT neutral | PARTIAL_NOT_PRIMARY | C05−C0 is neutral, but the C0≈P premise fails for pixel transfer. |
| C — AP harm | WEAK_HARM_NOT_MATERIAL | C05−C0 is slightly negative in most macro rows, at most 0.0114 pp; no material causal harm claim. |
| D — inference RMT harmful | NOT_PRIMARY | The small transport effect cannot explain the C0−P gap. |
| E — CIR training harms representation | SUPPORTED_FOR_MEDICAL_PIXEL_TRANSFER | C0 pixel AUROC/AP is below P at all six macro epochs; C05 does not recover. |
| F — training helps, inference does not | SOURCE_ONLY | C0 source AP improves, but that improvement does not transfer to Medical pixels. |
| G — training hurts, inference recovers | NOT_SUPPORTED | C05 remains below P on Medical pixel macros. |
| H — pixel good/image weak | NOT_PRIMARY | The corrected direction is mostly the reverse: image AUROC is stronger while pixel metrics are weaker. |
| I — image good/pixel AP weak | SUPPORTED | Image metrics are comparatively strong; Pixel AP is the clearest residual weakness. |
| J — domain flip | SUPPORTED_FOR_CIR_TRAINING | Liver improves while Retina/ClinicDB/Kvasir decline and Brain/ColonDB are mixed. |
| K — parent/upstream failure | NOT_PRIMARY | P is not universally weak and remains the stronger pixel-transfer reference, although the target problem is difficult. |

Formal CASE_A_K_DECISION.json preserves the same classifications and records the confidence limitation: no paired bootstrap CIs were run.

## Bottleneck fingerprint

BOTTLENECK_FINGERPRINT.json classifies the corrected state as:

- primary: K2_SOURCE_GOOD_TARGET_GENERALIZATION_BAD;
- secondary: K3_PIXEL_AUROC_GOOD_AP_BAD;
- measured risk: K7_TRAIN_DEPLOY_OPERATOR_MISMATCH;
- peer validity: structurally good but signal weak;
- MAD/saturation: bad in the bounded E14 Brain diagnostic;
- direct classification/segmentation conflict: not observed in the bounded batch;
- scheduler/optimizer/checkpoint state: good and matched.

## Gradient, loss, and deployment evidence

The exact corrected objective is cls_loss + seg_loss + 0.001 * kg_loss + 0.0 * k_loss. On one deterministic E14 VisA batch, classification loss was 0.3941401, segmentation loss 0.3562261, weighted KG 0.0004234, and weighted K 0. Classification/segmentation gradient cosine was +0.0338983; no direct negative conflict was observed. This is not a training-wide gradient proof.

The training map and deployment map differ: mean absolute difference 0.0032489, maximum 0.9993287, Pearson 0.6475441; deployment AP was 0.5963786 versus training-side AP 0.7237014 on the sampled batch. The mismatch is proven as a path difference, not proven as the cause of the Medical gap.

The bounded E14 Brain peer audit found valid K=8 peers, radius 3, no invalid/self/spatial/duplicate violations, and 0.107% post-hoc GT contamination. GT was diagnostic only. However, MAD median was 0.0038653 with minimum 3.22e-6; |delta|>0.99 covered 55.66% of diagnostic patches and |delta|>0.95 covered 63.52%. Stage 1 Pixel AP (0.4356) was much stronger than stage 0 (0.0745) and stage 2 (0.1023) on 60 selected images. These are reliability signals, not proof that a new transport rule will improve target metrics.

## Proven, correlational, unknown

### Proven in this snapshot

- The pre-fix CIR scheduler bug existed; the corrected pair now has matched StepLR state and learning-rate exposure.
- Both corrected training runs completed with valid finite candidate checkpoints.
- The exact P/C0/C05 source and Medical matrices completed.
- C0 is below P on Medical pixel macro metrics at every candidate epoch.
- C05−C0 is practically negligible in source and Medical macro results.
- The bounded train/deploy map mismatch and peer saturation/low-MAD behavior exist.

### Correlational or descriptive

- The association between corrected CIR training and the Medical transfer gap.
- The association between peer saturation/low MAD and domain-specific performance.
- The association between uneven stage attribution and Pixel AP.
- The association between the old checkpoint oscillation and excessive LR exposure; the old run was not a matched counterfactual.

### Unknown or incomplete

- paired bootstrap confidence intervals and sign consistency at image level;
- full corrected representation drift E10→E20;
- corrected normal-tail/anomaly quantiles and top-k precision;
- classification-only versus pixel-max-only image decomposition;
- full cross-epoch/cross-domain peer reliability;
- the causal share of train/deploy mismatch;
- whether an anchor or operator-consistency change improves held-out transfer.

## Research gate

RMT inference is not authorized for redesign: the current transport signal is not demonstrably useful, and C05 does not improve C0. No SAR-RMT, loss, optimizer, deployment, or architecture change is included. The diagnosis snapshot is complete; any future change must be a separate source-only, one-variable experiment.
