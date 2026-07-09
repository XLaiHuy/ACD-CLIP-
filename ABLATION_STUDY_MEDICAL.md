# Medical Transfer Ablation Summary

Protocol:

```text
Train source = VisA
Test target = 6 medical pixel-level datasets
Image-level mean uses 3 datasets only: BrainMRI, Liver CT, Retina OCT
Local protocol = pixel_stride=4, metric_thresholds=none unless noted
Metric format = AUC / AP
```

## Architecture Summary

| ID | Variant | Key architecture/setting |
|---|---|---|
| Mainbase | Local upstream-style baseline | Original ACD-CLIP local baseline, n_groups=3, no Phase1 DFG/SS2D changes |
| P1-V1 tau4 | Phase1 attention DFG | Replace MLP-style interaction with attention DFG, tau=4, no SS2D |
| P1-V2 tau8 | Phase1 attention DFG | Attention DFG, tau=8, no SS2D |
| P1-V3 SS2D | Phase1 attention + SS2D | Attention DFG tau=8 plus SS2D-style branch, gamma_max=0.2, feature_residual |
| P1-V3c old | Phase1 weight residual | Attention + SS2D, weight_residual, beta warmup 0.10, before fp32 attention stability fix |
| P1 final | Phase1 best anchor | V3c weight_residual, beta warmup 0.10, tau=8, n_groups=3, fp32 attention stability fix |
| P2B alpha0.1 K5e-3 | Hybrid hard-soft | alpha=0.1, lambda_kg=1e-2, lambda_k=5e-3 |
| P2B alpha0.2 K2e-3 | Hybrid hard-soft best pixel | alpha=0.2, lambda_kg=1e-2, lambda_k=2e-3 |
| P3B stagecons margin | Stage routing consistency | P2B alpha0.2 K2e-3 plus lambda_stage=5e-4, js_margin=0.02 |
| P3B stagecons warm bf16 | Stage routing consistency | P2B alpha0.2 K2e-3 plus lambda_stage=2e-3, JS no-margin, stage warmup 5 epochs, bf16 |

## Mean Results

| Variant | Epoch | 6-med pixel AUC/AP | 3-med image AUC/AP | Note |
|---|---:|---:|---:|---|
| Mainbase local strong | 10 | 90.25 / 38.21 | 73.80 / 74.36 | Chosen main baseline |
| P1-V1 tau4 | 15 | 87.45 / 31.47 | 73.04 / 74.07 | Attention DFG tau=4 |
| P1-V2 tau8 | 13 | 89.63 / 34.70 | 72.23 / 73.11 | Tau=8 improves vs tau=4 |
| P1-V3 SS2D | 11 | 90.23 / 34.77 | 73.23 / 72.58 | SS2D raises AUC, AP still weak |
| P1-V3c old | 9 | 90.91 / 35.63 | 72.10 / 73.02 | Weight residual before fp32 fix |
| P1 final best | 9 | 90.76 / 39.82 | 73.80 / 75.06 | Best balanced local anchor |
| P2B alpha0.1 K5e-3 | 7 | 90.52 / 37.81 | 72.70 / 73.03 | Stable but too conservative |
| P2B alpha0.2 K2e-3 | 10 | 90.98 / 40.35 | 71.65 / 70.71 | Best pixel-level mean so far |
| P3B stagecons margin | 7 | 90.27 / 37.93 | 73.26 / 72.23 | Negative ablation; stage loss inactive |
| P3B stagecons warm bf16 | 7 | 90.45 / 36.22 | 73.60 / 74.14 | Stable, but stage loss hurts Brain/colon transfer |

## Pixel-Level Per-Dataset Results

| Variant | Epoch | ColonDB | ClinicDB | Kvasir | BrainMRI | Liver CT | Retina OCT | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P1-V1 tau4 | 15 | 80.51 / 25.70 | 83.16 / 39.40 | 78.30 / 41.23 | 92.99 / 32.53 | 96.54 / 6.15 | 93.22 / 43.83 | 87.45 / 31.47 |
| P1-V2 tau8 | 13 | 81.11 / 28.56 | 86.89 / 47.82 | 87.96 / 59.31 | 93.73 / 27.11 | 95.59 / 4.05 | 92.53 / 41.36 | 89.63 / 34.70 |
| P1-V3 SS2D | 11 | 83.34 / 30.60 | 88.29 / 49.91 | 86.72 / 55.06 | 93.01 / 28.16 | 96.87 / 6.33 | 93.14 / 38.55 | 90.23 / 34.77 |
| P1-V3c old | 9 | 85.23 / 33.48 | 88.69 / 49.68 | 87.26 / 56.51 | 92.01 / 19.62 | 97.15 / 6.45 | 95.13 / 48.06 | 90.91 / 35.63 |
| P1 final best | 9 | 84.29 / 31.03 | 89.66 / 52.87 | 88.28 / 60.50 | 95.96 / 46.05 | 96.97 / 6.28 | 89.39 / 42.20 | 90.76 / 39.82 |
| P2B alpha0.1 K5e-3 | 7 | 83.74 / 34.52 | 90.76 / 58.53 | 87.11 / 57.93 | 95.55 / 40.26 | 95.41 / 4.19 | 90.54 / 31.45 | 90.52 / 37.81 |
| P2B alpha0.2 K2e-3 | 10 | 83.88 / 35.70 | 89.06 / 57.46 | 88.40 / 63.45 | 95.24 / 38.28 | 97.07 / 6.81 | 92.20 / 40.39 | 90.98 / 40.35 |
| P3B stagecons margin | 7 | 84.02 / 32.44 | 87.95 / 50.06 | 86.77 / 54.69 | 94.95 / 45.63 | 96.07 / 5.24 | 91.88 / 39.51 | 90.27 / 37.93 |
| P3B stagecons warm bf16 | 7 | 85.08 / 34.09 | 88.77 / 50.01 | 84.53 / 51.95 | 94.67 / 32.08 | 96.32 / 6.09 | 93.34 / 43.12 | 90.45 / 36.22 |

## Image-Level Per-Dataset Results

| Variant | Epoch | BrainMRI | Liver CT | Retina OCT | Mean |
|---|---:|---:|---:|---:|---:|
| Mainbase local strong | 10 | - | - | - | 73.80 / 74.36 |
| P1-V1 tau4 | 15 | 79.04 / 94.19 | 68.44 / 55.84 | 71.65 / 72.18 | 73.04 / 74.07 |
| P1-V2 tau8 | 13 | 81.93 / 94.62 | 57.26 / 49.06 | 77.50 / 75.65 | 72.23 / 73.11 |
| P1-V3 SS2D | 11 | 80.99 / 94.61 | 59.25 / 48.30 | 79.46 / 74.84 | 73.23 / 72.58 |
| P1-V3c old | 9 | 78.42 / 93.68 | 60.87 / 51.26 | 77.01 / 74.11 | 72.10 / 73.02 |
| P1 final best | 9 | 82.53 / 95.40 | 56.74 / 48.96 | 82.12 / 80.82 | 73.80 / 75.06 |
| P2B alpha0.1 K5e-3 | 7 | 84.17 / 95.91 | 58.47 / 52.89 | 75.47 / 70.28 | 72.70 / 73.03 |
| P2B alpha0.2 K2e-3 | 10 | 79.87 / 93.57 | 59.24 / 47.86 | 75.84 / 70.69 | 71.65 / 70.71 |
| P3B stagecons margin | 7 | 81.53 / 94.74 | 64.48 / 52.11 | 73.78 / 69.83 | 73.26 / 72.23 |
| P3B stagecons warm bf16 | 7 | 81.72 / 94.80 | 58.97 / 51.13 | 80.12 / 76.50 | 73.60 / 74.14 |

## Takeaways

```text
1. Phase1 final remains the best balanced pixel+image anchor.
2. Phase2B alpha0.2 K2e-3 is the best pixel-level mean so far: 90.98 / 40.35.
3. Phase2B gains come mostly from Colon/Kvasir/ClinicDB, while Brain image and Retina image suffer.
4. Phase3B js_margin=0.02 is a negative ablation because the stage loss was inactive.
5. Phase3B JS no-margin with warmup and bf16 is stable, but still underperforms Phase1/Phase2B on mean pixel AP.
6. The remaining Phase3B issue is not numerical overflow alone; q_ss2d_norm still drifts high, so future work needs score-scale control or a different routing loss.
```
