# Phase2B e10 Anchor Diagnosis Usage

This workflow is exploratory test-set ablation, not an unbiased final evaluation.

## Stage 1: e10 anchor diagnosis

Run:

```bash
cd /home/ai4/caohuy/ACD-CLIP-base-new-phase1
bash run_phase2b_e10_anchor_diagnosis.sh
```

Outputs:

```text
anchor_e10_diagnosis/image_score_raw_predictions.csv
anchor_e10_diagnosis/pixel_metrics_by_dataset.csv
anchor_e10_diagnosis/image_metrics_by_dataset.csv
anchor_e10_diagnosis/image_score_ablation_e10.csv
anchor_e10_diagnosis/anchor_best_config.txt
```

The raw CSV stores:

```text
dataset,epoch,prompt_config,file_name,label,cls_score,max_pixel,top1pct_pixel
```

`max_pixel` and `top1pct_pixel` are computed from the final anomaly map after blur,
interpolation, and softmax. `pixel_stride` is applied only to pixel AUC/AP.

## Stage 2: fixed-config epoch sweep

After choosing one config from Stage 1, run:

```bash
PROMPT_CONFIG=split_hard_cls \
SCORE_RULE=0.9_cls_0.1_top1pct \
bash run_phase2b_fixed_config_epoch_sweep.sh
```

The fixed sweep tests epochs 7-15 by default and writes:

```text
fixed_config_epoch_sweep.csv
```

Checkpoint selection:

```text
Primary: maximize pixel_ap_6
Constraint: image_ap_3 >= 73.80
Tie-break: image_ap_3
```

Strict success:

```text
pixel_ap_6 >= 40.20
image_ap_3 >= 74.50
```

Acceptable balance:

```text
pixel_ap_6 >= 39.82
image_ap_3 >= 73.80
```
