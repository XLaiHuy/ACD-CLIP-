# E8-reference Anchor evaluation

This is the controlled VisA-source `anchor_ref_e8_r1` run. Clean H was resumed through E8, its frozen image adapter was copied to `theta_ref_e8.pth`, and the Family-Safe Anchor continuation ran E9–E15. CIR was off throughout. E15 is the pre-specified primary epoch; earlier epochs below are diagnostics only.

## Macro trajectory

All metrics are percentages. Medical image metrics use the valid Brain/Liver/Retina scope only. Medical is raw-exact, pixel stride 1; MVTec is benchmark-exact, pixel stride 1.

| Epoch | Medical Pixel AUROC | Medical Pixel AP | Medical Image AUROC | Medical Image AP | MVTec Pixel AUROC | MVTec Pixel AP | MVTec Image AUROC | MVTec Image AP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E10 | 89.78 | 37.65 | 77.28 | 76.48 | 86.14 | 39.18 | 89.22 | 94.91 |
| E11 | 87.32 | 36.20 | 78.72 | 78.89 | 81.89 | 37.74 | 88.22 | 94.32 |
| E12 | 87.80 | 35.13 | 77.26 | 77.54 | 82.73 | 39.81 | 88.87 | 94.98 |
| E13 | 89.08 | 37.48 | 76.55 | 76.69 | 83.73 | 39.39 | 88.88 | 94.78 |
| E14 | 88.70 | 37.41 | 77.38 | 77.00 | 84.29 | 39.76 | 89.15 | 94.95 |
| **E15 primary** | **89.24** | **36.24** | **76.20** | **76.00** | **84.59** | **41.52** | **89.15** | **95.05** |

Machine-readable exact values are in [`medical_trajectory.csv`](medical_trajectory.csv) and [`mvtec_trajectory.csv`](mvtec_trajectory.csv).

## Medical per-dataset Pixel AP trajectory

| Dataset | E10 | E11 | E12 | E13 | E14 | E15 |
|---|---:|---:|---:|---:|---:|---:|
| ColonDB | 29.61 | 29.14 | 28.78 | 28.48 | 28.88 | 28.12 |
| ClinicDB | 48.82 | 47.91 | 49.30 | 46.92 | 45.01 | 45.39 |
| Kvasir | 52.42 | 51.35 | 52.61 | 51.27 | 48.52 | 51.81 |
| BrainMRI | 44.82 | 47.10 | 45.66 | 51.10 | 51.13 | 43.88 |
| Liver | 5.04 | 5.25 | 5.20 | 5.24 | 5.73 | 6.18 |
| Retina | 45.19 | 36.44 | 29.25 | 41.88 | 45.21 | 42.09 |

At E15, the Medical per-dataset Pixel AUROC/AP pairs are: ColonDB `83.62/28.12`, ClinicDB `86.30/45.39`, Kvasir `83.41/51.81`, BrainMRI `95.00/43.88`, Liver `96.42/6.18`, and Retina `90.69/42.09`. See [`medical_e15_dataset_metrics.csv`](medical_e15_dataset_metrics.csv) for the exact rows.

## MVTec E15 per-class Pixel metrics

| Class | Pixel AUROC | Pixel AP |
|---|---:|---:|
| bottle | 93.18 | 64.80 |
| cable | 78.21 | 28.57 |
| capsule | 93.07 | 35.07 |
| carpet | 98.21 | 78.51 |
| grid | 80.44 | 26.43 |
| hazelnut | 96.26 | 37.46 |
| leather | 99.22 | 58.86 |
| metal_nut | 28.26 | 12.05 |
| pill | 91.85 | 41.00 |
| screw | 97.95 | 16.78 |
| tile | 81.94 | 65.42 |
| transistor | 58.96 | 8.61 |
| toothbrush | 97.19 | 55.32 |
| wood | 86.77 | 54.28 |
| zipper | 87.30 | 39.71 |

Exact per-class values for every E10–E15 checkpoint are in [`mvtec_per_class_all_metrics.csv`](mvtec_per_class_all_metrics.csv); the Pixel-only export is [`mvtec_per_class_pixel.csv`](mvtec_per_class_pixel.csv).

## E15 deltas against frozen references

The H and Anchor-to-E1 reference values are the previously audited matched-protocol values in `/tmp/h2-thbr-evidence/audit/H2_4ARM_FINAL_PROTOCOL_AUDIT.md` and the associated result CSVs. Deltas are current E15 minus reference, in percentage points.

| Domain / comparison | Pixel AUROC | Pixel AP | Image AUROC | Image AP |
|---|---:|---:|---:|---:|
| Medical vs H E15 | -1.57 | +0.37 | +0.05 | -0.25 |
| Medical vs Anchor-to-E1 E15 | -2.01 | -3.22 | +0.92 | -0.35 |
| MVTec vs H E15 | -2.28 | -0.09 | -0.38 | -0.15 |
| MVTec vs Anchor-to-E1 E15 | -5.45 | -3.63 | -0.67 | +0.28 |

## Interpretation

- The Medical trajectory is not monotonic. E10 is the diagnostic peak for both macro pixel metrics, while E11 is the diagnostic peak for both valid-scope image metrics. E15 is reported as primary because it was pre-specified, not because it maximized a test metric.
- AUROC and AP answer different questions. AUROC is ranking quality across positive and negative pixels; AP is more sensitive to the positive-pixel base rate. Liver illustrates this directly: E15 has 96.42 Pixel AUROC but only 6.18 Pixel AP.
- Image and pixel metrics are not interchangeable. Image metrics classify whole samples and here average only Brain/Liver/Retina; pixel metrics evaluate localization over six datasets. The Medical E15 combination of 89.24 Pixel AUROC, 36.24 Pixel AP, 76.20 Image AUROC, and 76.00 Image AP should therefore be read as four separate summaries.
- Relative to Anchor-to-E1, the E8-reference continuation is lower on Medical and MVTec pixel metrics at E15, while its Medical image AUROC is higher and its MVTec image AP is slightly higher. Anchor-to-E1 had a longer continuation than Anchor-to-E8, so this comparison does not perfectly isolate reference age from anchor exposure duration; that limitation is material.

## Provenance and numerical notes

- Anchor reference file SHA256: `2af518565a1e81e85e3d06e6121d8636b1da51c522e1e0ec33f43c255be19e08`; internal reference SHA256: `27778e59f729a561ac9fee39213a9c748cef742db953e07cfcfa83840d9c8f22`.
- Anchor was active, CIR was off, and no H7/H8, resolution, multiscale, or other intervention was used.
- Training had zero non-finite losses. E14 had one non-finite-gradient skip; E9–E13 and E15 had zero such skips. All six Anchor checkpoints remain complete full-state checkpoints.
- Medical evaluation used evaluator commit `6bd932fbce0a425af5c8d3f7230dd7dc041568bd`, raw-exact stride 1, and the existing data root. PyTorch 2.6 required `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` to load trusted full-state checkpoints; this affects deserialization only. A temporary evaluator-checkout data symlink was removed after evaluation.
- The complete machine-readable provenance is [`provenance.json`](provenance.json), and the copied run manifest is [`run_manifest.json`](run_manifest.json).
