# Fresh source-transfer experiment results

Date: 2026-09-13 UTC
Branch: `experiment/fresh-source-rental-20260911`
Git SHA (implementation base): `d6b4e324832a577c1d537b87aa2e792361997fc6`

## Status

Complete. Both fresh source runs trained continuously from E1 through E20,
produced all twenty full-state checkpoints, completed Medical E1--E20
selection, and completed final Single-View plus locked Four-View TTA.

These are **TARGET-SELECTED EXPLORATORY TRANSFER EXPERIMENTS**: Medical
targets selected the checkpoint, so the transfer results are not untouched
target-only estimates.

| Run | Fresh source | Cross-domain target | Selected epoch | Medical macro Pixel AP | Medical macro Pixel AUROC |
|---|---|---|---:|---:|---:|
| V | VisA | MVTec AD | 7 | 36.6339995 | 88.2630985 |
| M | MVTec AD all supervised | VisA | 1 | 30.3784340 | 82.5589672 |

Selection used no TTA, `benchmark_exact`, `pixel_stride=1`, and the locked
rule: highest Medical macro Pixel AP; tie by higher Medical macro Pixel AUROC;
then earlier epoch. Run V E7 is clear over E8 (36.6339995 vs 35.7933742 AP).
Run M E1 is clear over E2 (30.3784340 vs 25.5281187 AP). TTA was not used
for selection.

## Final cross-domain target results

Values are percentages; delta is Four-View minus Single-View in percentage
points. Four-View uses the sequential reference implementation with
`identity,hflip,vflip,hvflip`.

| Run / target | Single pixel AP | Single pixel AUROC | Four-View pixel AP | Four-View pixel AUROC | Delta AP | Delta AUROC |
|---|---:|---:|---:|---:|---:|---:|
| V / MVTec (E7) | 40.042783 | 82.092745 | 42.928772 | 83.678209 | +2.885989 | +1.585464 |
| M / VisA (E1) | 22.170782 | 93.663529 | 24.279998 | 94.476186 | +2.109216 | +0.812657 |

Target image-level metrics were also recorded:

| Run / target | Single image AP / AUROC | Four-View image AP / AUROC | Delta AP / AUROC |
|---|---:|---:|---:|
| V / MVTec (E7) | 94.532783 / 88.415956 | 95.322878 / 90.180024 | +0.790095 / +1.764068 |
| M / VisA (E1) | 86.334339 / 83.489188 | 88.220020 / 85.392540 | +1.885681 / +1.903352 |

## Medical evaluation at the selected checkpoint

Pixel AP / pixel AUROC; delta is Four-View minus Single-View.

### Run V, selected E7

| Dataset | Single | Four-View | Delta |
|---|---:|---:|---:|
| Brain | 49.168185 / 95.077883 | 52.044236 / 96.512951 | +2.876051 / +1.435068 |
| Liver | 6.194192 / 94.549075 | 7.708334 / 96.372640 | +1.514142 / +1.823565 |
| Retina | 31.943776 / 88.373644 | 38.549759 / 86.126210 | +6.605983 / -2.247434 |
| Colon_clinicDB | 48.401237 / 87.439125 | 51.227225 / 88.781360 | +2.825988 / +1.342235 |
| Colon_colonDB | 30.928073 / 80.569022 | 34.237768 / 81.533673 | +3.309695 / +0.964651 |
| Colon_Kvasir | 53.168534 / 83.569842 | 55.850616 / 85.113590 | +2.682082 / +1.543748 |

### Run M, selected E1

| Dataset | Single | Four-View | Delta |
|---|---:|---:|---:|
| Brain | 52.059834 / 96.281352 | 54.389399 / 96.777334 | +2.329565 / +0.495982 |
| Liver | 6.724093 / 96.007605 | 6.030708 / 95.817390 | -0.693385 / -0.190215 |
| Retina | 47.616876 / 91.814248 | 40.404154 / 91.143711 | -7.212722 / -0.670537 |
| Colon_clinicDB | 25.621728 / 72.040488 | 27.686401 / 73.718478 | +2.064673 / +1.677990 |
| Colon_colonDB | 19.910787 / 71.231882 | 20.942548 / 73.330710 | +1.031761 / +2.098828 |
| Colon_Kvasir | 30.337286 / 67.978228 | 32.080393 / 69.967254 | +1.743107 / +1.989026 |

The evaluator does not support meaningful image-level metrics for the three
Colon datasets; their image AP/AUROC are reported as **N/A** here (the raw
evaluator logs encode them as zero). Brain, Liver, and Retina image metrics
are present in the JSON artifacts.

## Reproducibility and health checks

- Base model `ViT-L-14-336px.pt` SHA256:
  `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`.
- Python 3.11.16; torch 2.7.1+cu128; torchvision 0.22.1+cu128; CUDA 12.8;
  NVIDIA GeForce RTX 3090, 24 GB.
- `pytest -q`: 54 passed (15 warnings).
- Data validation passed: VisA 2162; MVTec all supervised 5354; Brain 3715;
  Liver 1493; Retina 1805; Colon_clinicDB 612; Colon_colonDB 380;
  Colon_Kvasir 1000.
- Preflight finished with `READY_TO_LAUNCH=YES`, anchor-only path confirmed,
  CUDA/AMP smoke pass, and E1-to-E20 continuation pass.
- Non-finite skip counts were V: 6 gradient skips, 0 loss skips; M: 9
  gradient skips, 0 loss skips. All were handled by the validated skip policy;
  neither run reached the abort threshold.
- Training wall time from the run logs was approximately V: 5:00:13 and M:
  12:31:14 (E1 fresh start through E20 completion).
- Run V: 20/20 checkpoints; E1 global step 359; E20 global step 7214;
  E20 scheduler last epoch 20. E1 SHA256:
  `5af82e501d3cc6fb5a4bb60e93c5ce2614b08bbde54888d05272c7dac645b3db`.
  E20 SHA256:
  `087834c38722ae45755ae3a74cee86c626856861fed069390bcfb4db47a48ed2`.
- Run M: 20/20 checkpoints; E1 global step 891; E20 global step 17851;
  E20 scheduler last epoch 20. E1 SHA256:
  `733d67f0a5d82d69cda29ab5023d3802b70579f22c275954194c69582a9a67f4`.
  E20 SHA256:
  `e8699a4c808759475834f032ebf333720529a0c4f6a7af1231407ddc2fb8687d`.
- Both runs used the fixed fresh-E1 Family-Safe reference,
  `anchor_lambda=0.0021633926715180626`, family budget `rho=0.10`, complete
  family partitions, and recorded finite anchor telemetry. Run M's final
  maximum active-family effective ratio was `0.0999999813`.
- Both final evaluation orchestrations contain
  `SELECTED_CHECKPOINT_EVALUATION_COMPLETE`; Four-View outputs are finite.
- The safe sequential TTA parity gate passed. Batched TTA candidates were not
  enabled because strict parity showed small numerical map/metric differences;
  the locked sequential reference was used for the reported results.

## Artifact locations

The compact, Git-tracked result and provenance files are in `handoff/rental_20260911/compact_results/`. They are direct copies of the authoritative runtime outputs below.

The complete uncommitted runtime outputs remain in the workspace:

- Run V training: `runs/fresh_visa_source_20260911/`
- Run V selection: `results/rental_20260911/run_v_medical_selection/`
- Run V final evaluation: `results/rental_20260911/run_v_selected_epoch7/`
- Run M training: `runs/fresh_mvtec_all_source_20260911/`
- Run M selection: `results/rental_20260911/run_m_medical_selection/`
- Run M final evaluation: `results/rental_20260911/run_m_selected_epoch1/`

Each final evaluation directory contains `environment.txt`,
`orchestration.log`, Single-View logs, `four_view_tta.json`, and
`DELTA_TTA.json`. Checkpoints and pixel-score spools are intentionally not
part of the Git commit.
