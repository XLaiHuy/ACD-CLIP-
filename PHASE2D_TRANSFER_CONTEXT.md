# Phase2C/Phase2D Transfer Context

## Publication

- Repository: `https://github.com/XLaiHuy/ACD-CLIP-.git`
- Publication branch: `phase2d-medical-ready`
- Phase2C canonical branch/commit: `phase2c` / `eb3e2319242b9a14df8337eba481ad783f02972e`
- Phase2D LB base: `phase2d-lb-0p1` / `b17a057d811253987b8605bf764fcffb3e8254a8`
- Phase2D AB context: `phase2d-ab-interpolation` / `40239faf5c71e499b24f70cb21a66e562c15deb9`
- AB history is already an ancestor of the LB publication base.

## Published Phase2D checkpoints

See `checkpoints/phase2d/PHASE2D_CHECKPOINT_MANIFEST.json` for the machine-readable manifest.

| State | Repository path | Epoch/lambda | SHA-256 | Bytes | Pixel AUC/AP | Image AUC/AP |
|---|---|---:|---|---:|---:|---:|
| LB_0p1 | `checkpoints/phase2d/LB_0p1_seed42/LB_0p1_seed42_e15_pixelAUC97.4206_pixelAP53.4980_imageAP98.2721.pth` | e15 | `cf59bcbed5e00d60bfcbc9955ffb16928ccd887d97b03d53774eed237bce922d` | 56451506 | 97.4206 / 53.4980 | 97.7083 / 98.2721 |
| AB25 | `checkpoints/phase2d/AB_interpolation_seed42/AB25_lambdaB0p25.pth` | e13 / .25 | `e9f6d3339b8d6766b2227ed718fa05c8a0effcc407343dffbe8a5557ea7cbff7` | 56451586 | 95.5693 / 55.4652 | 97.9653 / 98.4647 |
| AB50 | `checkpoints/phase2d/AB_interpolation_seed42/AB50_lambdaB0p50.pth` | e13 / .50 | `610b8e9a89d339dfc7893a006cf9a848d99e218a4abfef3da8b6576a2442d824` | 56451586 | 96.0225 / 55.4166 | 97.8611 / 98.3772 |
| AB75 | `checkpoints/phase2d/AB_interpolation_seed42/AB75_lambdaB0p75.pth` | e13 / .75 | `606f312654c70317db10bb2d5d137643d53ffa409cbed065e0eb87b7427d37ce` | 56451586 | 96.1600 / 55.2958 | 97.9792 / 98.5016 |

## Phase2C status

- A-prime e13 is already materialized and verified on `phase2c`.
- B e13 is already materialized and verified on `phase2c`.
- PL/P-LoRA-only e15 is already materialized and verified on `phase2c`; it remains distinct from full P.
- C e14 was not present on the GPU and is expected to be recovered from the lab machine.
- P/full PCGrad e13 was not present on the GPU and is expected to be recovered from the lab machine.
- No P/PL substitution was made.

## Configuration and protocol

- Dataset: VisA fixed split, seed 42.
- Backbone: OpenAI CLIP ViT-L/14-336.
- Image size: 518.
- `n_groups=3`.
- Phase2D LB: A-prime configuration, hybrid alpha max 0.20, beta warm-up to 0.10, selected epoch 15.
- AB25/AB50/AB75: epoch-13 A-prime/B interpolation with lambda B 0.25/0.50/0.75.
- AB parent hashes: A-prime `036143f9ff940716684174e569ca07a8a060a9b81de94c14e8ba49d748783752`; B `b556a2083555b1b9a2d29050b515808d191f224832613a203a90b74f5847cc2d`.
- Medical evaluator: `phase2b_anchor_diagnosis.py`.
- Medical settings: batch size 8, workers 6, pixel stride 4, metric thresholds None, TTA off, `cls_only` image score.
- Medical map: Gaussian sigma 1.5, kernel 9x9, bilinear final resize, `align_corners=True`, abnormal channel 1.
- Pixel datasets: Brain, Liver, Retina, Colon_clinicDB, Colon_colonDB, Colon_Kvasir.
- Image datasets: Brain, Liver, Retina only.
- Dataset files and pretrained model are not included in this publication branch.
- No medical result was used to reselect a checkpoint.

## Lab continuation

```bash
git clone https://github.com/XLaiHuy/ACD-CLIP-.git
cd ACD-CLIP-
git switch phase2d-medical-ready
git lfs pull
sha256sum checkpoints/phase2d/LB_0p1_seed42/*.pth checkpoints/phase2d/AB_interpolation_seed42/*.pth
```

Phase2C A-prime, B and PL can be obtained from `phase2c` at commit `eb3e2319242b9a14df8337eba481ad783f02972e`. C and P remain lab-machine expected artifacts.
