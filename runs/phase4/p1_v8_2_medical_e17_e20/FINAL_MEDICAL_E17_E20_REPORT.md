# P1-v8.2 Six-Medical Evaluation Final Report (Epochs 17 & 20)

**Repository**: `/home/ai4/caohuy/ACD-CLIP-phase4`  
**Branch**: `phase4-progress1-cops-dynamic-prompt`  
**Commit HEAD**: `96c5b9c6ad8ec2b3b2eaec11a5b0deab58d41b2c`  
**Evaluation Split**: Official 6 Medical Test Split (`split=test`)  
**Protocol Status**: `MEDICAL_E17_E20_COMPLETED`  

> [!IMPORTANT]
> **Epoch 20** is the canonical final checkpoint.
> **Epoch 17** is POST-HOC DIAGNOSTIC TRAJECTORY reporting only (NOT model selection).

## 1. Two-Epoch Medical Summary Table

| Epoch | Pixel AUROC Macro | Pixel AP Macro | Image AUROC Macro | Image AP Macro | Support-Aware Combined Score | Status |
|---|---|---|---|---|---|---|
| 17 | 87.51% | 34.30% | 71.83% | 72.29% | 66.01% | DIAGNOSTIC TRAJECTORY |
| 20 | 87.86% | 34.11% | 72.23% | 73.07% | 66.31% | **CANONICAL FINAL** |

## 2. Per-Dataset Breakdown (Epochs 17 & 20)

### Epoch 17 (Diagnostic)

| Dataset | Pixel AUROC | Pixel AP | Image AUROC | Image AP | Image Metric Support |
|---|---|---|---|---|---|
| Brain | 91.72% | 27.25% | 79.19% | 93.55% | Supported (Valid) |
| Liver | 95.86% | 5.55% | 55.75% | 47.56% | Supported (Valid) |
| Retina | 85.89% | 35.46% | 80.55% | 75.76% | Supported (Valid) |
| Colon_clinicDB | 85.59% | 47.55% | N/A | N/A | Unsupported (Excluded) |
| Colon_colonDB | 79.87% | 30.25% | N/A | N/A | Unsupported (Excluded) |
| Colon_Kvasir | 86.14% | 59.71% | N/A | N/A | Unsupported (Excluded) |

### Epoch 20 (Canonical Final)

| Dataset | Pixel AUROC | Pixel AP | Image AUROC | Image AP | Image Metric Support |
|---|---|---|---|---|---|
| Brain | 92.25% | 26.89% | 79.25% | 93.41% | Supported (Valid) |
| Liver | 96.21% | 6.10% | 56.70% | 48.30% | Supported (Valid) |
| Retina | 87.17% | 35.77% | 80.73% | 77.51% | Supported (Valid) |
| Colon_clinicDB | 85.51% | 48.01% | N/A | N/A | Unsupported (Excluded) |
| Colon_colonDB | 80.10% | 29.70% | N/A | N/A | Unsupported (Excluded) |
| Colon_Kvasir | 85.91% | 58.19% | N/A | N/A | Unsupported (Excluded) |

