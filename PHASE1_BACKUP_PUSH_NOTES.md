# Phase 1 Backup Push Notes

Date: 2026-06-29

Local backup branch:

```text
backup/phase1-v3c-fp32attn-20260629
```

Target GitHub repo:

```text
https://github.com/XLaiHuy/ACD-CLIP-.git
```

Remote name added locally:

```text
huy
```

## Best Current Result

Best local Phase 1 checkpoint:

```text
phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3/adapter_9.pth
```

Best epoch:

```text
epoch 9
```

Medical zero-shot results:

```text
Pixel mean over 6 datasets: 90.76 / 39.82
Image mean over 3 datasets: 73.80 / 75.06
```

Protocol:

```text
METRIC_THRESHOLDS=none
PIXEL_STRIDE=4
n_groups=3
img_size=518
dfg_mode=attn
dfg_attn_dim=256
dfg_attn_tau=8.0
dfg_ss2d_fusion=weight_residual
dfg_beta_schedule=warmup010
dfg_beta_target=0.10
dfg_weight_residual_fp32=True
```

## Important Push Note

Git LFS is not installed on this lab machine. The checkpoint folders contain
many `.pth` files and are several GB total. Pushing all checkpoints with normal
Git may be slow and may make the GitHub repo very heavy.

Recommended minimum to preserve the work:

```text
Commit and push code, scripts, dataset jsonl files, experiment logs, test logs,
and the best checkpoint adapter_9.pth from the fixed fp32 attention run.
```

If full raw reproducibility is required, install Git LFS first or archive/copy
the checkpoint folders outside Git.

## Useful Commands

Push current branch after committing:

```bash
git push -u huy backup/phase1-v3c-fp32attn-20260629
```

Run best fixed V3c test again:

```bash
bash test_phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3_selected_epochs.sh 9
```

Train fixed V3c again:

```bash
bash train_phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3.sh
```
