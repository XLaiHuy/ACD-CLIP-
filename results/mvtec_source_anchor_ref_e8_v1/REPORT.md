# MVTec-source E8-reference Anchor evaluation

This is the source-dataset reversal of the controlled E8-reference Anchor protocol. The original `dataset/hub/MVTec.jsonl` was used directly as the MVTec training source: fresh source E1, clean H resumed through E8, frozen E8 image adapter `theta_ref`, then Anchor E9–E15. CIR was off throughout. The resulting checkpoints were evaluated on VisA at E10–E15 with benchmark-exact, pixel-stride-1 settings. E15 is the pre-specified primary epoch; no target-based epoch selection was used.

## Macro trajectory

All metrics are percentages. The VisA rows are macro averages over its 12 target categories.

| Epoch | VisA Pixel AUROC | VisA Pixel AP | VisA Image AUROC | VisA Image AP |
|---:|---:|---:|---:|---:|
| E10 | 93.24 | 23.11 | 83.47 | 86.00 |
| E11 | 90.89 | 16.01 | 80.68 | 84.62 |
| E12 | 91.29 | 16.26 | 80.56 | 84.23 |
| E13 | 91.28 | 17.22 | 81.79 | 85.00 |
| E14 | 89.95 | 16.11 | 81.44 | 85.01 |
| **E15 primary** | **92.74** | **18.39** | **81.46** | **85.07** |

Exact values are in [`visa_trajectory.csv`](visa_trajectory.csv). E10 is the diagnostic maximum for all four macro metrics, but E15 is reported as primary because it was pre-specified.

## VisA E15 per-class summary

| Class | Pixel AUROC | Pixel AP | Image AUROC | Image AP |
|---|---:|---:|---:|---:|
| candle | 96.38 | 27.45 | 93.49 | 93.73 |
| pcb3 | 81.94 | 4.32 | 72.54 | 73.21 |
| capsules | 95.73 | 20.61 | 81.30 | 89.25 |
| pipe_fryum | 93.31 | 16.54 | 97.78 | 98.93 |
| pcb4 | 93.11 | 16.94 | 90.56 | 90.58 |
| macaroni2 | 94.17 | 0.28 | 61.05 | 64.38 |
| pcb2 | 85.30 | 4.14 | 72.17 | 76.96 |
| chewinggum | 98.12 | 74.88 | 95.76 | 98.02 |
| macaroni1 | 95.63 | 1.47 | 77.60 | 78.15 |
| cashew | 94.91 | 22.83 | 55.68 | 72.61 |
| fryum | 93.53 | 24.59 | 87.98 | 93.94 |
| pcb1 | 90.70 | 6.60 | 91.65 | 91.06 |

Exact all-metric rows for E10–E15 are in [`visa_per_class_all_metrics.csv`](visa_per_class_all_metrics.csv); the pixel-only export is [`visa_per_class_pixel.csv`](visa_per_class_pixel.csv).

## Interpretation

- The reversal transfers well in pixel AUROC at E15 (92.74) but has much lower pixel AP (18.39). This is consistent with AP's sensitivity to the positive-pixel base rate and the sparse/small defect regions in VisA; AUROC and AP should not be treated as interchangeable.
- Performance is class-dependent. E15 pixel AP is strongest for chewinggum (74.88), while macaroni2 (0.28), macaroni1 (1.47), pcb2 (4.14), and pcb3 (4.32) are the main localization bottlenecks. Whole-image AP remains substantially higher for those classes, showing that image-level recognition and pixel localization separate here.
- The trajectory is non-monotonic: E10 is the diagnostic peak, E11–E14 decline or fluctuate, and E15 partially recovers. This is why the pre-specified E15 endpoint is retained rather than selecting E10 from target performance.
- There is no valid numerical delta table against the previously frozen H E15 and Anchor-to-E1 E15 references for this reversal: those references are VisA-source runs evaluated on Medical/MVTec, whereas this run is MVTec-source evaluated on VisA. Comparing those cross-direction/domain aggregates would conflate source and target changes. The complementary VisA-source matched-reference deltas are recorded in [`../anchor_ref_e8_r1/REPORT.md`](../anchor_ref_e8_r1/REPORT.md).
- As with the complementary experiment, an Anchor-to-E1 continuation was not run for this reversal. Thus this result establishes the MVTec-source E8-reference Anchor point, not a clean isolation of reference age from anchor exposure duration.

## Provenance and numerical notes

- Manifest audit: [`MANIFEST_AUDIT.md`](MANIFEST_AUDIT.md). The original MVTec manifest SHA256 is `3a5e304ea16bba82e6e525d188698e91ca92b718696f8c257ed435d235b4cc2c`; the pool is exactly 1,725 records: 467 normal and 1,258 anomaly, across all 15 categories. Missing images/masks, duplicates, train/good records, anomaly-without-mask records, and unexpected normal masks were all zero.
- Source training used ViT-L-14-336, image size 518, three groups, LoRA rank 16/alpha 2, Conv-LoRA rank 8/alpha 2 with kernels 3/5, HPA-DSR, four hybrid soft-prompt tokens, Anchor lambda `0.0021633926715180626`, family budget rho `0.10`, and CIR off. Optimizer settings were image LR `0.001`, text LR `0.0005`, soft LR `0.00005`, StepLR gamma `0.9`, gradient clip `1.0`, batch size 6, seed 0.
- Clean-H training had non-finite-gradient skips: E1 `1`, E2 `2`, E3–E8 `0`. Anchor training had E9 `0`, E10 `0`, E11 `0`, E12 `1`, E13 `1`, E14 `0`, E15 `0`. Non-finite-loss skips were zero for every epoch. All 15 checkpoints passed full-state, finite-tensor, configuration, and provenance audits.
- The E8 checkpoint SHA256 is `69fe1382c8240d37dca9acb8ac9cc7e10fe10a3c4d6285af7df37cb80fc48fce`. The frozen theta reference file SHA256 is `c5482dbde2d50144f6c12459bbc72781d3c34bf37f2cadfc505bdb9cfec08373`; its internal reference SHA256 is `d815ea099b1e019fc8508c854c7d2b77b1eebf88742a28666c94342d9ae63a90`.
- VisA evaluation used manifest SHA256 `468463d2d6234fa7537c6da32b027758527676a12a54a4028c5a282cdd726842`, `test.py` benchmark-exact mode, pixel stride 1, and the evaluator code at commit `1f7426f18a311db4b572329048bb191f151f323a`. PyTorch trusted-checkpoint deserialization used `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`; this changes loading behavior only, not the metric protocol.
- Complete machine-readable provenance is in [`provenance.json`](provenance.json) and the copied run manifest is [`run_manifest.json`](run_manifest.json). The external frozen run root is `/workspace/mvtec_source_anchor_ref_e8_v1_run`.
