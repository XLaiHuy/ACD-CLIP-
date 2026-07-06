# Phase2A Prompt Weight KL Shared From Scratch

## Branch / Commit

- Branch: `phase2a_prompt_weight_kl_fromscratch`
- Implementation commit: `2bad415 implement phase2a shared prompt weighting`
- Base organized Phase1 commit: `500ee88 organize phase1 repo layout`

## Intent

Phase2A tests whether replacing the old prompt-template mean aggregation with shared learned prompt weights improves zero-shot medical transfer.

This run is from scratch on VisA. It does not load a Phase1 checkpoint.

## Main Change

Only prompt aggregation is changed:

```text
Phase1:
  normalized prompt embeddings -> mean -> normalize prototype

Phase2A:
  normalized prompt embeddings -> shared weighted mean -> normalize prototype
```

Shared prompt weights:

```text
raw_w_normal   [6]
raw_w_abnormal [10]
```

No group-wise prompt weights, no KgCoOp, no uniform mix, no new prompts.

## Phase1 Settings Kept

```text
dataset=VisA
n_groups=3
dfg_mode=attn
dfg_attn_dim=256
dfg_attn_tau=8.0
use_ss2d_dfg=True
dfg_gamma_max=0.2
dfg_ss2d_fusion=weight_residual
dfg_beta=0.10
dfg_beta_schedule=warmup010
dfg_beta_target=0.10
text_adapt_weight=0.2
batch_size=6
epoch=20
amp=True
grad_checkpointing=True
num_workers=6
non_finite_loss_abort_threshold=5
```

Prompt weighting settings:

```text
prompt_weight_lr=5e-5
prompt_weight_lambda_kl=1e-4
prompt_weight_temperature=2.0
prompt_weight_freeze_epochs=3
```

Prompt LR was fixed so epochs 1-3 keep `5e-5` while prompt weights are frozen. Epoch 4 is the first effective prompt-weight update at `5e-5`.

## Run Artifact Locations

Local artifacts were not committed to git:

```text
runs/phase2a/phase2a_prompt_weight_kl_shared_fromscratch/train.log
runs/phase2a/phase2a_prompt_weight_kl_shared_fromscratch/test.log
runs/phase2a/phase2a_prompt_weight_kl_shared_fromscratch/adapter_1.pth ... adapter_15.pth
runs/phase2a/phase2a_prompt_weight_kl_shared_fromscratch/nonfinite_diagnostics/
```

Reason: checkpoints/log directories are bulky experiment artifacts. This report records the context needed to interpret the branch.

## Train Summary

Train loss decreased cleanly until epoch 15:

```text
epoch 1   mean_loss=1.3918   mean_seg_loss=0.7083
epoch 5   mean_loss=1.1563   mean_seg_loss=0.5252
epoch 9   mean_loss=0.8969   mean_seg_loss=0.4437
epoch 12  mean_loss=0.8215   mean_seg_loss=0.4128
epoch 15  mean_loss=0.7707   mean_seg_loss=0.3800
```

Non-finite loss remained zero through epoch 15. Epoch 16 aborted:

```text
RuntimeError: non_finite_loss=6 exceeded threshold 5 at epoch 16
latest diagnostic:
runs/phase2a/phase2a_prompt_weight_kl_shared_fromscratch/nonfinite_diagnostics/epoch_016_batch_00163_skip_0006.pth
```

Diagnostic evidence:

```text
epoch_text_features finite=True
prompt_kl finite=True
prompt weights finite and near uniform

seg_features finite=False
det_features finite=False
cls_pred finite=False
seg_pred finite=False

stage1 DFG finite
stage2 DFG finite
stage3 DFG/SS2D NaN
```

The checkpoint `adapter_15.pth` was finite:

```text
image_adapter: finite
text_adapter: finite
prompt_weighting: finite
```

## Prompt Weight Behavior

Prompt weights stayed very close to uniform:

```text
normal max:
0.1667 -> 0.1679

abnormal max:
0.1000 -> 0.1012
```

Prompt KL remained tiny:

```text
epoch 15 prompt_kl = 5.63e-5
lambda = 1e-4
```

Interpretation: Phase2A v1 did not learn meaningfully different prompt prototypes, but it still changed the optimization trajectory.

## Brain Test Gate

Only Brain was tested as a quick gate before spending time on all 6 medical datasets.

Phase2A Brain results:

```text
epoch 8   pixel 93.02 / 37.06   image 77.59 / 93.42
epoch 9   pixel 94.64 / 39.14   image 79.07 / 93.83
epoch 10  pixel 94.42 / 38.07   image 78.81 / 94.10
epoch 11  pixel 94.43 / 37.79   image 80.22 / 94.67
epoch 12  pixel 93.80 / 34.51   image 79.06 / 94.44
epoch 13  pixel 94.21 / 33.35   image 80.91 / 94.81
epoch 14  pixel 93.50 / 28.90   image 78.43 / 94.00
```

Phase1 best Brain reference:

```text
epoch 9   pixel 95.96 / 46.05   image 82.53 / 95.40
```

Best Phase2A v1 Brain epoch was epoch 9:

```text
pixel 94.64 / 39.14
image 79.07 / 93.83
```

Compared with Phase1 best:

```text
pixel AUROC -1.32
pixel AP    -6.91
image AUROC -3.46
image AP    -1.57
```

## Conclusion

Phase2A v1 is not promising enough to test all 6 medical datasets.

Observed failure mode:

```text
train loss improves on VisA
Brain zero-shot AP drops clearly
epoch 16 image/DFG stage3 becomes numerically unstable
```

Likely interpretation:

```text
shared prompt weighting v1 changes the train trajectory without creating a useful enough prompt prototype shift.
The image/DFG branch then drifts into a less stable and less transferable region.
```

This does not look like source code contamination from old DORA/other phase experiments. The image branch implementation was unchanged relative to Phase1 best; Phase2A only added prompt-weighting machinery and train/test compatibility.

## Recommended Next Step

Do not continue Phase2A v1 as the main direction.

If Phase2A is continued, use a new run only as a bounded probe:

```text
phase2a_prompt_weight_kl_shared_scoreclamp80_lr1e4

Changes:
1. DFG attention score fp32 + clamp [-80, 80]
2. prompt_weight_lr 5e-5 -> 1e-4

Gate:
test Brain epochs 8-12 only.
If Brain pixel AP does not approach >=44, stop Phase2A.
```

Otherwise, move to Phase2B from the Phase1 best organized commit:

```bash
git switch -c phase2b_from_phase1_best 500ee88
```

