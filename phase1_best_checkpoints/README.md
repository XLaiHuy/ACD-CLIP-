# Checkpoints By Phase

This folder is the clean, human-readable backup layout.

## Recommended Final Anchor

```text
phase1b_v3c_fp32attn_final/e09_best_final_anchor_adapter.pth
```

Result:

```text
Pixel mean, 6 medical datasets: 90.76 / 39.82
Image mean, 3 medical datasets: 73.80 / 75.06
```

## Folder Map

| Folder | Meaning |
|---|---|
| `baseline_mainbase/` | Original local ACD-CLIP baseline checkpoints from `ACD-CLIP-mainbase-newest` |
| `phase1a_v1_tau4/` | Phase 1A V1, dual-softmax DFG, tau=4 |
| `phase1a_v2_tau8/` | Phase 1A V2, dual-softmax DFG, tau=8 |
| `phase1b_v3_ss2d_g02/` | Phase 1B V3, feature_residual SS2D-style branch, gamma_max=0.2 |
| `phase1b_v3c_betawarm010_old/` | V3c weight_residual beta warmup before fp32 attention stability fix |
| `phase1b_v3c_fp32attn_final/` | V3c weight_residual beta warmup with fp32 attention stability fix, final Phase 1 anchor |

## Checkpoint Map

| File | Epoch | Result / use |
|---|---:|---|
| `baseline_mainbase/e09_best_pixel_adapter.pth` | 9 | Baseline best pixel AP, pixel mean 89.77 / 38.24, image mean 73.33 / 73.89 |
| `baseline_mainbase/e10_best_pixel_auc_adapter.pth` | 10 | Baseline best pixel AUC, pixel mean 90.25 / 38.21, image mean 73.80 / 74.36 |
| `phase1a_v1_tau4/e15_best_pixel_adapter.pth` | 15 | V1 tau4 best pixel, pixel mean 87.45 / 31.47 |
| `phase1a_v1_tau4/e17_best_image_adapter.pth` | 17 | V1 tau4 best image, image mean 73.38 / 74.60 |
| `phase1a_v2_tau8/e13_best_pixel_adapter.pth` | 13 | V2 tau8 best pixel, pixel mean 89.63 / 34.70 |
| `phase1a_v2_tau8/e10_best_image_adapter.pth` | 10 | V2 tau8 best image, image mean 72.65 / 73.58 |
| `phase1b_v3_ss2d_g02/e11_best_pixel_adapter.pth` | 11 | V3 SS2D g02 best pixel, pixel mean 90.23 / 34.77 |
| `phase1b_v3_ss2d_g02/e12_best_image_adapter.pth` | 12 | V3 SS2D g02 best image, image mean 74.08 / 74.24 |
| `phase1b_v3c_betawarm010_old/e09_best_old_before_fp32_adapter.pth` | 9 | Old V3c betawarm010 before fp32 attention fix, pixel mean 90.91 / 35.63 |
| `phase1b_v3c_fp32attn_final/e09_best_final_anchor_adapter.pth` | 9 | Best local Phase 1 final anchor, pixel mean 90.76 / 39.82, image mean 73.80 / 75.06 |

## Restore Example

To restore final anchor:

```bash
cp phase1b_v3c_fp32attn_final/e09_best_final_anchor_adapter.pth \
  /home/ai4/caohuy/ACD-CLIP-base-new-phase1/phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3/adapter_9.pth
```

Then test:

```bash
cd /home/ai4/caohuy/ACD-CLIP-base-new-phase1
bash test_phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3_selected_epochs.sh 9
```
