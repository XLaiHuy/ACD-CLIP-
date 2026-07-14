# Phase 2C + Phase 2D: Full Experiment Comparison

## Scope and metric convention

- Dataset: **VisA**, fixed split `seed42`.
- Validation macro row: `n=432`.
- Metrics are percentages; higher is better.
- `Pixel AUC/AP`: pixel-level localization/segmentation quality.
- `Image AUC/AP`: image-level anomaly classification quality.
- Checkpoint selection follows each run's registered `selection.json`; it is not necessarily the epoch with the largest value of every metric.
- All runs use `cls_only` image scoring unless noted otherwise.

## 1. Main comparison: selected checkpoints

| Phase | Version / run | Main setting | Epoch | Pixel AUC | Pixel AP | Image AUC | Image AP | Decision |
|---|---|---|---:|---:|---:|---:|---:|---|
| 2C | A-prime / A_alpha020 | Hybrid alpha max 0.20 | 13 | **94.8038** | **55.5341** | **97.9028** | **98.4225** | Primary winner before Phase 2D |
| 2C | B / B_alpha015 | Hybrid alpha max 0.15 | 13 | 96.2236 | 55.1342 | 97.8750 | 98.4287 | Higher Pixel AUC; lower Pixel AP |
| 2C | C / C_alpha020_delayed | Alpha and soft prompt delayed | 14 | 96.1128 | 54.7353 | 97.1944 | 97.6479 | Not selected |
| 2C | P / P_pcgrad | PCGrad on 4 shared groups | 13 | **97.1696** | 51.7660 | 96.3819 | 96.7979 | Fails Pixel-AP-first rule |
| 2C | PL / P_LoRA_only | PCGrad on `shared_image_lora` only; batch 8 | 15 | 96.6840 | 52.7478 | 97.3542 | 97.9956 | Exploratory; PCGrad branch closed |
| 2D | LB_0p1 | A-prime rerun; LB 0.1 protocol | 15 | **97.4206** | 53.4980 | 97.7083 | **98.2721** | Phase 2D LB checkpoint |
| 2D | AB25 | A-prime/B interpolation, lambda B=0.25 | 13 | 95.5693 | 55.4652 | 97.9653 | 98.4647 | Secondary Pareto candidate |
| 2D | AB50 | A-prime/B interpolation, lambda B=0.50 | 13 | 96.0225 | 55.4166 | 97.8611 | 98.3772 | Secondary Pareto candidate |
| 2D | AB75 | A-prime/B interpolation, lambda B=0.75 | 13 | 96.1600 | 55.2958 | **97.9792** | **98.5016** | Secondary Pareto candidate |

## 2. Phase 2C settings

Common settings for A-prime, B and C:

- OpenAI CLIP ViT-L/14-336, resized to image size 518.
- Conv-LoRA image adaptation, attention DFG + SS2D residual-weight fusion.
- BF16, 15 epochs, batch size 6, 6 workers, seed 42.
- `dfg_beta=0.10`, beta warm-up to 0.10.
- `lambda_kg=0.01`, `lambda_k=0.002`.
- `image_lr=1e-3`, `text_lr=5e-4`, `soft_prompt_lr=5e-5`.
- Four-token hybrid soft prompt, frozen for the first 3 epochs.

| Version | Distinguishing setting | Alpha schedule | Selection constraint | Selected checkpoint |
|---|---|---|---|---|
| A-prime | Reference configuration | `0,0,0,0.05,0.10,0.20,...` | Image AP >= 92.4348 | `adapter_13.pth` |
| B | Lower hybrid alpha, max 0.15 | `0,0,0,0.0375,0.075,0.15,...` | Image AP >= 92.4348 | `adapter_13.pth` |
| C | Delayed activation; soft prompt freeze 5 epochs | `0,0,0,0,0,0.05,0.10,0.20,...` | Image AP >= 92.4348 | `adapter_14.pth` |
| P | Deterministic symmetric two-task PCGrad on `shared_image_lora`, `m_i_w`, `hard_text_adapter`, `soft_prompt` | Same as A-prime | Run-local guardrail | `adapter_13.pth` |
| PL | PCGrad only on `shared_image_lora`; batch size 8, 10 workers | Same as A-prime | Run-local guardrail | `adapter_15.pth` |

### Phase 2C deltas versus A-prime

| Version | Pixel AUC delta | Pixel AP delta | Image AUC delta | Image AP delta |
|---|---:|---:|---:|---:|
| B | +1.4198 | -0.3999 | -0.0278 | +0.0062 |
| C | +1.3090 | -0.7988 | -0.7083 | -0.7746 |
| P | +2.3658 | -3.7682 | -1.5208 | -1.6246 |
| PL | +1.8802 | -2.7863 | -0.5486 | -0.4269 |

Interpretation: PCGrad increases Pixel AUC but consistently damages Pixel AP. A-prime remains the best Pixel-AP-first reference.

## 3. Phase 2D settings and results

### LB_0p1

LB_0p1 is an A-prime configuration rerun with the registered Phase 2D LB protocol. It uses the same main architecture and schedule as A-prime: alpha max 0.20, beta warm-up to 0.10, BF16, batch size 6, 15 epochs, and `cls_only` scoring. The selected checkpoint is epoch 15.

### A-prime/B interpolation

The parent reproduction gate passed within **0.05 percentage points** for every registered macro metric. The three locked candidates interpolate the A-prime and B adapter states at epoch 13:

| Candidate | Lambda B | Pixel AUC | Pixel AP | Image AUC | Image AP | Primary eligible? |
|---|---:|---:|---:|---:|---:|---|
| AB25 | 0.25 | 95.5693 | 55.4652 | 97.9653 | 98.4647 | No |
| AB50 | 0.50 | 96.0225 | 55.4166 | 97.8611 | 98.3772 | No |
| AB75 | 0.75 | 96.1600 | 55.2958 | 97.9792 | 98.5016 | No |

All three candidates are secondary Pareto candidates, but none exceeds the primary Pixel-AP threshold while preserving the registered image guardrail. Therefore the Phase 2D interpolation decision is: **keep A-prime as primary winner**.

## 4. Ranking by metric

| Metric | Best run | Value |
|---|---|---:|
| Pixel AUC | LB_0p1 | 97.4206 |
| Pixel AP | A-prime | 55.5341 |
| Image AUC | AB75 | 97.9792 |
| Image AP | AB75 | 98.5016 |

The metric winners are not identical. For the preregistered Pixel-AP-first objective, A-prime remains the canonical reference; AB75 is an attractive exploratory trade-off with the best Image AP and higher Pixel AUC than A-prime, but slightly lower Pixel AP.

## 5. Final conclusion

1. **Canonical Phase 2C winner:** A-prime, epoch 13.
2. **Best Pixel-AUC result:** Phase 2D LB_0p1, epoch 15.
3. **Best Image-AUC/Image-AP result:** Phase 2D AB75.
4. **Best balanced result under the registered Pixel-AP-first rule:** A-prime.
5. All comparisons are single-seed (`seed42`); multi-seed confirmation is required before claiming robustness.

## Artifact locations

- Phase 2C A/B/C/P: `runs/phase2c_bf16/<run>/visa_val_metrics.csv`
- Phase 2C PL: `runs/phase2c_4090/PL_lora_only_seed42_bs8/visa_val_metrics.csv`
- Phase 2D LB: `runs/phase2d_lb_0p1_seed42/visa_val_metrics.csv`
- Phase 2D interpolation: `runs/phase2d_ab_interpolation_seed42/candidate_metrics.csv`
- Selection records: each run's `selection.json`
