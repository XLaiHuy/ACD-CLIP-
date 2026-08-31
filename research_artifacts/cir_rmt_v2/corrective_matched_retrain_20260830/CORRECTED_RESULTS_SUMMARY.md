# Corrected matched CIR-V2 results

## Technical summary

The corrected matched experiment completed successfully: parent Phase2B (P) and corrected CIR training both finished 20 epochs with candidates E10/E12/E14/E16/E18/E20. Source evaluation completed 18/18 cells and Medical evaluation completed 108/108 cells.

The scheduler confound is removed. Parent and CIR have identical post-step learning rates, Adam state shape, StepLR state, global optimizer steps, FP32 policy, prompt schedule, loss, and checkpoint timing at every candidate. The original CIR scheduler bug is therefore no longer part of this comparison.

The scientific result is not an inference-RMT win. Corrected CIR-native deployment (C0) remains below matched P on Medical Pixel AUROC and Pixel AP at every macro epoch; C05−C0 is effectively zero. The remaining signal is most consistent with a CIR training/representation transfer problem, not a clean causal failure of the RMT idea as an inference operator.

Current decision: PHASE2B_REPRESENTATION_PRESERVATION.

## Definitions and comparison basis

- P: matched Phase2B control checkpoint.
- C0: the same corrected CIR checkpoint with native alpha=0 deployment.
- C05: the same corrected CIR checkpoint with alpha=0.5 RMT deployment.
- TRAIN_EFFECT = C0 - P.
- RMT_INFERENCE_EFFECT = C05 - C0.
- TOTAL_CIR_EFFECT = C05 - P.

Metrics are unweighted macro means across the six Medical targets. Pixel metrics have six-target support. Image metrics have three-target support because the frozen evaluator defines Colon image metrics as undefined. Values below are percentage points.

## Medical macro decomposition

| epoch | Pixel AUROC P | Pixel AUROC C0 | Pixel AUROC C05 | train | RMT inference | Pixel AP P | Pixel AP C0 | Pixel AP C05 | train | RMT inference |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E10 | 87.9118 | 86.4998 | 86.4920 | -1.4120 | -0.0079 | 31.8093 | 29.9577 | 29.9528 | -1.8516 | -0.0049 |
| E12 | 88.7800 | 86.8187 | 86.8083 | -1.9614 | -0.0103 | 32.7435 | 30.1531 | 30.1444 | -2.5904 | -0.0087 |
| E14 | 88.2263 | 86.8275 | 86.8215 | -1.3988 | -0.0060 | 32.3308 | 31.0151 | 31.0137 | -1.3158 | -0.0013 |
| E16 | 88.1549 | 87.0886 | 87.0771 | -1.0663 | -0.0114 | 32.2745 | 30.8630 | 30.8627 | -1.4115 | -0.0003 |
| E18 | 88.4967 | 87.7208 | 87.7200 | -0.7759 | -0.0008 | 32.9389 | 30.6427 | 30.6482 | -2.2961 | +0.0055 |
| E20 | 87.9457 | 87.2548 | 87.2532 | -0.6908 | -0.0016 | 32.7511 | 30.2431 | 30.2414 | -2.5079 | -0.0017 |

Image-level metrics show a different phenotype:

| epoch | Image AUROC P | Image AUROC C0 | Image AUROC C05 | train | RMT inference | Image AP P | Image AP C0 | Image AP C05 | train | RMT inference |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E10 | 69.9125 | 71.6897 | 71.6892 | +1.7772 | -0.0004 | 72.2193 | 71.7143 | 71.7151 | -0.5049 | +0.0008 |
| E12 | 70.2226 | 70.5355 | 70.5361 | +0.3130 | +0.0006 | 72.4045 | 72.0790 | 72.0782 | -0.3255 | -0.0008 |
| E14 | 67.7974 | 71.2141 | 71.2142 | +3.4167 | +0.0001 | 70.4857 | 72.5429 | 72.5428 | +2.0573 | -0.0002 |
| E16 | 68.1336 | 70.8425 | 70.8438 | +2.7089 | +0.0013 | 71.0013 | 72.0803 | 72.0816 | +1.0790 | +0.0013 |
| E18 | 68.4045 | 70.9639 | 70.9636 | +2.5593 | -0.0003 | 70.9687 | 72.1165 | 72.1165 | +1.1477 | +0.0001 |
| E20 | 68.9410 | 69.9275 | 69.9273 | +0.9865 | -0.0003 | 70.9636 | 72.1828 | 72.1826 | +1.2192 | -0.0002 |

The exact per-target/per-epoch values, checkpoint identities, and undefined image metrics are in corrected_medical_decomposition.csv.

## Source decomposition

Corrected C0 source Pixel AP is above P at every candidate epoch (51.3210→63.3897% for P versus 56.2201→63.9643% for C0). C0 source Pixel AUROC is slightly lower through E14, essentially matched at E16, and above P at E20. C05−C0 is tiny at all source epochs. Thus source adaptation quality does not predict the Medical pixel transfer gap in this run.

## Domain pattern

Mean C0−P over E10–E20, in percentage points:

| target | Pixel AUROC | Pixel AP | interpretation |
|---|---:|---:|---|
| Brain | +0.519 | -1.914 | AUROC slightly higher, AP weaker |
| Liver | +2.337 | +4.554 | relative success/control domain |
| Retina | -3.134 | -7.306 | AUROC and AP loss; AP-specific weakness |
| Colon_clinicDB | -3.692 | -5.876 | strong transfer failure |
| Colon_colonDB | -0.664 | +1.918 | mixed metric direction |
| Colon_Kvasir | -2.672 | -3.350 | systematic loss |

This is a domain flip in CIR training, not evidence for domain-specific alpha tuning. C05 follows C0 almost exactly.

## Historical pre-fix evidence

The old buggy-trained run remains preserved under forensics_20260830_pre_scheduler_fix/. Its completed six-Medical pixel macro was E12 89.0550/35.3933, E14 89.5530/35.5607, E16 87.9140/32.0348, E18 89.4628/34.2885, and E20 86.9373/30.9680 AUROC/AP percentage. The published ACD-CLIP N=3 reference recorded there is 91.55/43.03. The earlier E14→E16 drop→E18 recovery→E20 drop, Liver relative success, Retina AUROC-good/AP-weak pattern, Brain AP weakness, and Colon/Kvasir underperformance remain historical observations, but the scheduler bug prevents clean RMT attribution.

## Evidence limits

The matrix is descriptive and paired by checkpoint/method, but image-level paired bootstrap confidence intervals were not run. The snapshot therefore treats sub-0.02 percentage-point C05−C0 effects as practically neutral by magnitude and mixed sign, not as a formal null test. Full corrected representation drift, AP-tail quantiles, and classification-only/pixel-max-only image decomposition were not rerun.
