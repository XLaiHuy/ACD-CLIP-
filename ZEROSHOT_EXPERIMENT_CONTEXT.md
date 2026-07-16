# Zero-shot Experiment Context and Phase 3 Handoff

Date: 2026-07-16
Branch snapshot: `zeroshot-complete`
Previous working branch: `phase2d-medical-ready`

## 1. Purpose

This file consolidates the zero-shot experiment state before starting Phase 3 few-shot work. It records:

- architecture variants tested;
- experiment scripts and result locations;
- medical zero-shot metrics;
- VisA Phase 2C/2D metrics;
- checkpoint policy and required artifacts;
- recommended branch/phase naming for the next work.

The recommended interpretation is:

- Phase 1 = medical zero-shot architecture ablations.
- Phase 2C/2D = VisA zero-shot adaptation / interpolation experiments.
- `zeroshot-complete` = frozen branch containing the zero-shot evidence and required checkpoints.
- Phase 3 should start as few-shot work from this frozen zero-shot branch.

## 2. Branch and naming decision

Use a new branch instead of renaming the existing Phase 2D branch.

Recommended branch layout:

| Branch | Role |
|---|---|
| `phase2d-medical-ready` | Historical working branch for Phase 2D medical-ready experiments. |
| `zeroshot-complete` | Frozen complete zero-shot snapshot with reports, metrics, scripts, and required checkpoints. |
| `phase3-fewshot` | New future branch for few-shot experiments, branched from `zeroshot-complete`. |

Rationale:

- Renaming `phase2d-medical-ready` would obscure the historical meaning of the existing branch.
- `zeroshot-complete` is clearer as the boundary between zero-shot and few-shot work.
- Phase 3 can start cleanly from `zeroshot-complete` without mixing new few-shot changes into zero-shot evidence.

## 3. Architecture variants tested

All medical Phase 1 variants use OpenAI CLIP ViT-L/14-336 with image size 518, `n_groups=3`, Conv-LoRA image adaptation, and hard prompt evaluation unless otherwise stated.

| Variant | Short name | Main setting | Notes |
|---|---|---|---|
| Phase 1A V1 | `v1_attn_tau4` | Attention DFG, tau=4 | Baseline attention-DFG ablation. |
| Phase 1A V2 | `v2_attn_tau8` | Attention DFG, tau=8 | Stronger zero-shot baseline than tau=4. |
| Phase 1B V3 | `v3_attn_tau8_ss2d_feature_g02` | Attention DFG + SS2D `feature_residual`, gamma max 0.2 | Improves Pixel AUC over V2 but lower Pixel AP. |
| Phase 1B V3c old | `v3c_attn_tau8_ss2d_weightres` | Attention DFG + SS2D `weight_residual`, beta warmup 0.10 | Best average Pixel AUC among listed medical Phase 1 runs. |
| Phase 1B V3c final | `v3c_weightres_fp32attn_g3` | Weight residual + fp32 attention stability fix | Best average Pixel AP; recommended Phase 1 medical anchor. |
| MLP+SS2D ablation | `mlp_ss2d_feature_g02` | MLP DFG + SS2D `feature_residual`, gamma max 0.2 | Useful ablation; improves Colon AP but fails Retina. |

Implementation context for the MLP+SS2D ablation:

- `model/adapter.py` now permits `use_ss2d_dfg` for `dfg_mode="mlp"` only when `dfg_ss2d_fusion="feature_residual"`.
- The SS2D residual is added to the MLP gate input before the group softmax.
- `test.py` validates MLP checkpoints against `use_ss2d_dfg`, `dfg_gamma_max`, and `dfg_ss2d_fusion`.

## 4. Medical zero-shot comparison

Metric convention:

- Metrics are percentages; higher is better.
- Best checkpoint per dataset is selected by Pixel AUC for the table below.
- Image metrics are only meaningful for Brain, Liver, Retina. Colon image metrics are `0.00` because image-level labels/metrics are not available or not computed for these colon sets.

### 4.1 Average comparison across six medical datasets

| Phase | Avg Pixel AUC | Avg Pixel AP | Avg Image AUC, nonzero only | Avg Image AP, nonzero only |
|---|---:|---:|---:|---:|
| `v1_attn_tau4` | 87.68 | 31.77 | 73.33 | 74.46 |
| `v2_attn_tau8` | 90.42 | 35.76 | 72.22 | 72.52 |
| `v3_attn_tau8_ss2d_feature_g02` | 90.67 | 34.29 | 73.31 | 73.13 |
| `v3c_attn_tau8_ss2d_weightres` | 90.91 | 35.63 | 72.10 | 73.02 |
| `v3c_weightres_fp32attn_g3` | 90.83 | 40.07 | 73.10 | 74.22 |
| `mlp_ss2d_feature_g02` | 88.17 | 34.37 | 69.43 | 72.53 |

Interpretation:

- Best Pixel AUC: `v3c_attn_tau8_ss2d_weightres`, 90.91.
- Best Pixel AP: `v3c_weightres_fp32attn_g3`, 40.07.
- Recommended medical zero-shot anchor: `v3c_weightres_fp32attn_g3`, because Pixel AP is the most materially improved localization metric and the average Pixel AUC is essentially tied with the best.
- `mlp_ss2d_feature_g02` is not recommended as the main model. It is retained as an ablation because it improves Colon AP but degrades Retina strongly.

### 4.2 MLP+SS2D versus baseline V1

| Dataset | V1 Pixel AUC/AP | MLP+SS2D Pixel AUC/AP | Delta AUC | Delta AP |
|---|---:|---:|---:|---:|
| Brain | 92.99 / 32.53 | 93.03 / 35.06 | +0.04 | +2.53 |
| Liver | 96.76 / 6.06 | 97.21 / 7.29 | +0.45 | +1.23 |
| Retina | 93.64 / 44.40 | 86.41 / 25.07 | -7.23 | -19.33 |
| Colon_clinicDB | 83.16 / 39.40 | 87.87 / 52.66 | +4.71 | +13.26 |
| Colon_colonDB | 80.62 / 26.79 | 80.62 / 33.75 | +0.00 | +6.96 |
| Colon_Kvasir | 78.91 / 41.47 | 83.88 / 52.36 | +4.97 | +10.89 |

MLP+SS2D conclusion:

- Improves over V1 on average Pixel AP.
- Strong on Colon localization.
- Not robust because Retina drops sharply.
- Should be reported as a partial-positive / negative ablation, not as the main architecture.

### 4.3 Per-dataset best checkpoints for MLP+SS2D

| Dataset | Best epoch by Pixel AUC | Pixel AUC | Pixel AP | Image AUC | Image AP |
|---|---:|---:|---:|---:|---:|
| Brain | 8 | 93.03 | 35.06 | 73.22 | 92.66 |
| Liver | 9 | 97.21 | 7.29 | 62.56 | 53.68 |
| Retina | 8 | 86.41 | 25.07 | 72.51 | 71.26 |
| Colon_clinicDB | 8 | 87.87 | 52.66 | 0.00 | 0.00 |
| Colon_colonDB | 8 | 80.62 | 33.75 | 0.00 | 0.00 |
| Colon_Kvasir | 8 | 83.88 | 52.36 | 0.00 | 0.00 |

Required MLP+SS2D checkpoint policy:

- Keep epoch 8 as the primary MLP+SS2D ablation checkpoint because it is best for most datasets and best average behavior.
- Keep epoch 9 only to preserve the Liver-best checkpoint.
- Do not upload all 20 MLP+SS2D training checkpoints.

## 5. VisA zero-shot Phase 2C/2D comparison

Scope:

- Dataset: VisA, fixed split `seed42`.
- Metrics are validation macro percentages.
- Scoring: `cls_only` unless noted otherwise.
- Phase 2C/2D are zero-shot transfer/adaptation experiments, not few-shot Phase 3.

| Phase | Run | Main setting | Epoch | Pixel AUC | Pixel AP | Image AUC | Image AP | Decision |
|---|---|---|---:|---:|---:|---:|---:|---|
| 2C | A-prime | Hybrid alpha max 0.20 | 13 | 94.8038 | 55.5341 | 97.9028 | 98.4225 | Canonical Pixel-AP-first winner |
| 2C | B | Hybrid alpha max 0.15 | 13 | 96.2236 | 55.1342 | 97.8750 | 98.4287 | Higher Pixel AUC, slightly lower Pixel AP |
| 2C | C | Alpha and soft prompt delayed | 14 | 96.1128 | 54.7353 | 97.1944 | 97.6479 | Not selected |
| 2C | P | PCGrad on multiple shared groups | 13 | 97.1696 | 51.7660 | 96.3819 | 96.7979 | Pixel AUC up, Pixel AP damaged |
| 2C | PL | PCGrad on shared image LoRA only | 15 | 96.6840 | 52.7478 | 97.3542 | 97.9956 | Exploratory; PCGrad branch closed |
| 2D | LB_0p1 | A-prime rerun with LB 0.1 protocol | 15 | 97.4206 | 53.4980 | 97.7083 | 98.2721 | Best Pixel AUC |
| 2D | AB25 | A/B interpolation, lambda B=0.25 | 13 | 95.5693 | 55.4652 | 97.9653 | 98.4647 | Secondary Pareto |
| 2D | AB50 | A/B interpolation, lambda B=0.50 | 13 | 96.0225 | 55.4166 | 97.8611 | 98.3772 | Secondary Pareto |
| 2D | AB75 | A/B interpolation, lambda B=0.75 | 13 | 96.1600 | 55.2958 | 97.9792 | 98.5016 | Best Image AUC/AP tradeoff |

Phase 2C/2D conclusion:

- Canonical zero-shot VisA winner under the Pixel-AP-first rule: A-prime, epoch 13.
- Best Pixel AUC: LB_0p1, epoch 15.
- Best Image AUC/AP: AB75.
- PCGrad increases Pixel AUC but consistently damages Pixel AP, so it should not be the default path.
- A/B interpolation creates Pareto candidates but does not beat A-prime on the registered primary objective.

## 6. Medical evaluation of Phase 2C/2D checkpoints

Medical Phase 2C/2D evaluation files are under `medical_phase2cd_results/`.

| Config | Epoch | Pixel AUC, six datasets | Pixel AP, six datasets | Image AUC, three datasets | Image AP, three datasets |
|---|---:|---:|---:|---:|---:|
| A_prime | 13 | 88.56 | 34.73 | 75.63 | 76.43 |
| B | 13 | 90.18 | 36.35 | 76.16 | 76.02 |
| C | 14 | 89.81 | 35.07 | 75.10 | 75.01 |
| P | 13 | 89.00 | 35.64 | 72.87 | 74.86 |
| PL | 15 | 90.91 | 36.74 | 74.13 | 76.09 |
| LB_0p1 | 15 | 90.55 | 34.13 | 71.95 | 72.59 |
| AB25 | 13 | 89.30 | 35.65 | 75.04 | 75.66 |
| AB50 | 13 | 89.81 | 36.28 | 75.45 | 75.70 |
| AB75 | 13 | 90.08 | 36.43 | 75.49 | 75.48 |

Medical transfer interpretation:

- `PL` is best by medical Pixel AUC/AP among Phase 2C/2D evaluated checkpoints.
- `B` is best by medical Image AUC.
- `A_prime` is best by medical Image AP.
- Phase 2C/2D VisA winners do not map perfectly to medical winners, so Phase 3 few-shot should validate on target medical data directly.

## 7. Artifact and script map

### 7.1 Medical Phase 1 logs and scripts

| Artifact | Purpose |
|---|---|
| `phase1_v1_attn_tau4/test.log` | V1 baseline medical test log. |
| `phase1_v2_attn_tau8/test.log` | V2 tau8 medical test log. |
| `phase1_v3_attn_tau8_ss2d_g02/test.log` | V3 feature residual SS2D medical test log. |
| `phase1_v3c_attn_tau8_ss2d_weightres_betawarm010/test.log` | V3c weight residual medical test log. |
| `phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3/test.log` | Final Phase 1 medical anchor test log. |
| `phase1_mlp_ss2d_feature_g02/test.log` | MLP+SS2D ablation test log. |
| `train_phase1_mlp_ss2d_feature_g02.sh` | MLP+SS2D training entrypoint. |
| `test_phase1_mlp_ss2d_feature_g02_selected_epochs.sh` | MLP+SS2D selected-epoch evaluation entrypoint. |
| `run_phase1_mlp_ss2d_feature_g02_train_then_test.sh` | MLP+SS2D train-then-test wrapper. |

### 7.2 Phase 2C/2D logs and scripts

| Artifact | Purpose |
|---|---|
| `PHASE2C_2D_ALL_RESULTS.md` | Full VisA Phase 2C/2D comparison. |
| `PHASE2D_TRANSFER_CONTEXT.md` | Transfer context from Phase 2D. |
| `phase2cd_medical_eval.py` | Medical evaluation script for Phase 2C/2D checkpoints. |
| `run_phase2cd_medical_eval.sh` | Runs medical evaluation over Phase 2C/2D checkpoints. |
| `medical_phase2cd_results/*/macro_metrics.csv` | Macro medical metrics per Phase 2C/2D checkpoint. |
| `medical_phase2cd_results/*/dataset_metrics.csv` | Dataset-level medical metrics per Phase 2C/2D checkpoint. |

## 8. Required checkpoint inventory

Do not push full training directories or every epoch checkpoint. Keep only final/selected adapter checkpoints.

### 8.1 Phase 1 medical checkpoints

| Checkpoint | Role |
|---|---|
| `phase1_best_checkpoints/baseline_mainbase/e09_best_pixel_adapter.pth` | Mainbase baseline best Pixel AP. |
| `phase1_best_checkpoints/baseline_mainbase/e10_best_pixel_auc_adapter.pth` | Mainbase baseline best Pixel AUC. |
| `phase1_best_checkpoints/phase1a_v1_tau4/e15_best_pixel_adapter.pth` | V1 best pixel checkpoint. |
| `phase1_best_checkpoints/phase1a_v1_tau4/e17_best_image_adapter.pth` | V1 best image checkpoint. |
| `phase1_best_checkpoints/phase1a_v2_tau8/e13_best_pixel_adapter.pth` | V2 best pixel checkpoint. |
| `phase1_best_checkpoints/phase1a_v2_tau8/e10_best_image_adapter.pth` | V2 best image checkpoint. |
| `phase1_best_checkpoints/phase1b_v3_ss2d_g02/e11_best_pixel_adapter.pth` | V3 best pixel checkpoint. |
| `phase1_best_checkpoints/phase1b_v3_ss2d_g02/e12_best_image_adapter.pth` | V3 best image checkpoint. |
| `phase1_best_checkpoints/phase1b_v3c_betawarm010_old/e09_best_old_before_fp32_adapter.pth` | V3c old checkpoint before fp32 attention fix. |
| `phase1_best_checkpoints/phase1b_v3c_fp32attn_final/e09_best_final_anchor_adapter.pth` | Recommended final medical zero-shot anchor. |
| `phase1_best_checkpoints/phase1_mlp_ss2d_feature_g02/e08_best_majority_pixel_adapter.pth` | MLP+SS2D ablation checkpoint for most datasets / average behavior. |
| `phase1_best_checkpoints/phase1_mlp_ss2d_feature_g02/e09_best_liver_adapter.pth` | MLP+SS2D Liver-best checkpoint. |

MLP+SS2D checkpoint checksums:

```text
9a9808265d3e45053ca77919c9ab2efb4857f6327941a79d085a5fde014e2114  phase1_best_checkpoints/phase1_mlp_ss2d_feature_g02/e08_best_majority_pixel_adapter.pth
2de3526fb6bd9041150ed874d34811867e7a96ddfc5ab984b1fb9705673da67d  phase1_best_checkpoints/phase1_mlp_ss2d_feature_g02/e09_best_liver_adapter.pth
```

### 8.2 Phase 2C/2D checkpoints

| Checkpoint | Role |
|---|---|
| `checkpoints/phase2c/A_prime_seed42/A_prime_seed42_e13_pixelAUC94.8038_pixelAP55.5341_imageAP98.4225.pth` | Canonical Phase 2C Pixel-AP-first winner. |
| `checkpoints/phase2c/B_seed42/B_seed42_e13_pixelAUC96.2236_pixelAP55.1342_imageAP98.4287.pth` | A-prime comparison / parent for interpolation. |
| `checkpoints/phase2c/PL_lora_only_seed42/PL_seed42_bs8_e15_pixelAUC96.6840_pixelAP52.7478_imageAP97.9956.pth` | PL medical-transfer reference. |
| `checkpoints/phase2d/LB_0p1_seed42/LB_0p1_seed42_e15_pixelAUC97.4206_pixelAP53.4980_imageAP98.2721.pth` | Best VisA Pixel AUC. |
| `checkpoints/phase2d/AB_interpolation_seed42/AB25_lambdaB0p25.pth` | A/B interpolation candidate. |
| `checkpoints/phase2d/AB_interpolation_seed42/AB50_lambdaB0p50.pth` | A/B interpolation candidate. |
| `checkpoints/phase2d/AB_interpolation_seed42/AB75_lambdaB0p75.pth` | Best VisA Image AUC/AP tradeoff. |

Do not upload or duplicate `model/ViT-L-14-336px.pt` unless explicitly required. It is a large pretrained CLIP weight file and should be treated as an external dependency, not an experiment artifact.

## 9. Phase 3 few-shot recommendation

Start Phase 3 from `zeroshot-complete`.

Recommended first Phase 3 branch:

```bash
git switch zeroshot-complete
git switch -c phase3-fewshot
```

Recommended Phase 3 starting checkpoints:

1. Medical few-shot target: `phase1_best_checkpoints/phase1b_v3c_fp32attn_final/e09_best_final_anchor_adapter.pth`
2. VisA/industrial transfer few-shot target: `checkpoints/phase2c/A_prime_seed42/A_prime_seed42_e13_pixelAUC94.8038_pixelAP55.5341_imageAP98.4225.pth`
3. If optimizing Pixel AUC instead of Pixel AP: `checkpoints/phase2d/LB_0p1_seed42/LB_0p1_seed42_e15_pixelAUC97.4206_pixelAP53.4980_imageAP98.2721.pth`
4. If optimizing image-level AP/AUC: `checkpoints/phase2d/AB_interpolation_seed42/AB75_lambdaB0p75.pth`

Do not use `mlp_ss2d_feature_g02` as the Phase 3 default. Keep it only as an ablation or as a Colon-focused hypothesis.
