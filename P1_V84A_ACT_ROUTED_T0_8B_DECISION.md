# P1-v8.4-A routed ACT T0 smoke decision

Source commit: `a9cdb4b67d3613d8b8c1428be6ef4f661d382c05`
Artifact: `runs/p1_v84a_gpu/act_routed_t0_8b_seed0/smoke_summary.json`

## Locked run

- Architecture: P1-v8.4-A; teacher object `g_route`.
- `act_gain_threshold=0.0`; `lambda_act=7.435420936678605e-05`.
- `lambda_factor=0.03`, `lambda_router=0.10`; factor/router tau `0.05`;
  Router gain threshold `0.02`; entropy threshold `0.98`; fixed rho `0.05`.
- Fresh OpenAI CLIP-only initialization, seed 0, VisA/train, image 518,
  batch 1, accumulation 6, FP32, AMP/TF32 off, gradient checkpointing on.
- Successful launch used `/workspace/.venv-p1v84a/bin/python train.py` with
  `--h6_smoke_max_batches 8` and milestones `1 2 3 4 5 6 7 8`.
- The exact parsed full configuration is retained in
  `runs/p1_v84a_gpu/act_routed_t0_8b_seed0/config.json`; the scientific launch
  controls were:

  ```text
  --dataset VisA --img_size 518 --epoch 1 --batch_size 1 --grad_accum_steps 6
  --precision fp32 --grad_checkpointing --h6_progress 1
  --h6_progress_version P1-v8.4-A --h6_prediction_routing dense
  --h6_global_text_mode phase2b_hybrid --use_hybrid_soft_prompt
  --h6_local_factor_mode center_spread --h6_local_center_mix 0.05
  --h6_local_factor_spread 0.10 --lambda_h6_factor 0.03
  --lambda_h6_router 0.10 --lambda_h6_act 7.435420936678605e-05
  --h6_act_gain_threshold 0.0 --h6_act_effective_beta 0.999
  --h6_utility_factor_effective_beta 0.999 --h6_router_support_normalized
  --h6_factor_tau_utility 0.05 --h6_router_tau_utility 0.05
  --h6_router_gain_threshold 0.02 --h6_utility_entropy_threshold 0.98
  --h6_primary_anchored_factor_surgery --h6_collect_router_gradient_geometry
  --h6_drift_diagnostics --h6_factor_grad_diagnostics
  --h6_trajectory_milestones 1 2 3 4 5 6 7 8 --h6_smoke_max_batches 8
  ```
- Exactly 8/8 microbatches and 2 optimizer steps. No checkpoint is retained;
  the generated local adapter was moved out of the run directory and was not
  staged.

## ACT support and trajectory

The zero-boundary labels were reconstructed exactly: all positive/negative/
ambiguous mismatch counts are zero, and ambiguous support is zero.

| scope | ON | OFF | ambiguous | valid |
|---|---:|---:|---:|---:|
| overall | 13,415 (45.2689%) | 16,219 (54.7311%) | 0 | 29,634 |
| normal | 13,406 (45.2569%) | 16,216 (54.7431%) | 0 | 29,622 |
| anomaly | 9 (75.0%) | 3 (25.0%) | 0 | 12 |

Per-batch overall ON/OFF counts were:

`1: 2187/1485, 2: 2619/1200, 3: 910/2066, 4: 2320/1787, 5: 3054/729,
6: 2252/1243, 7: 14/4093, 8: 59/3616`.

ACT probability was exactly `0.5` through the first accumulation window,
`0.5028887987` on the post-step-1 probe, and `0.5028770566` on batch 8
(batch-8 range `0.49683678..0.51278782`). The output head moved from zero
(`weight norm 0 -> 0.0027643966` after step 1); upstream ACT gradient after
that update was `0.00056063995`. The compact original runtime record did not
retain conditional ON/OFF probability means, so their numeric separation is
not claimed; final teacher AUROC was `0.52432424` (>0.5), a directional proxy.

## Utility and gradient diagnostics

| region | Base | FullSoftRouted_ACT1 | ActualGated | mean `g_route` |
|---|---:|---:|---:|---:|
| overall | 0.6331883454 | 0.6332190910 | 0.6332039125 | -0.0001650045 |
| normal | 0.6331681386 | 0.6331989085 | 0.6331837216 | -0.0001650854 |
| anomaly | 0.6830682755 | 0.6830438375 | 0.6830560565 | 0.0000345800 |

The actual-run total-ACT-head/shared-main gradient proxy had raw
median/p95/max `0.70069346/1.04660808/1.08504304` and weighted
median/p95/max `0.0000520995/0.0000778197/0.0000806775`. The weighted p95 and
max are within the safety envelope (`<=0.5`, `<=1.0`). The exact isolated
main-only versus ACT-only runtime vectors were not persisted by the original
smoke logger; the fresh no-step calibration reference remains raw
`4901.8242/6724.5688/6934.4507` and weighted
`0.3644713/0.5/0.5156056` (median/p95/max).

## Mechanical contract

- No NaN/Inf; finite gradients and trainable parameters on every batch/step.
- Residual definition, routed correction, ActualGated reconstruction, surgery
  reconstruction, and MAIN exact-change maximum errors were all exactly `0`.
- Both optimizer windows reported fixed rho `[0.0500000007, 0.0500000007,
  0.0500000007]`, with rho non-trainable.
- No Router formulation, threshold search, loss reweighting, or capacity
  change was made.

## Decision

All mechanical, zero-boundary semantic, support, ACT-head, and gradient-safety
checks pass. This is a mechanically healthy ACT-only smoke, not authorization
for a 300-batch run.

Decision code: **`ACT_8B_MECHANICALLY_HEALTHY`**

Next authorized action: discussion only regarding an ACT-only fresh 8B follow-up;
do not launch 300B automatically. No Router or capacity experiment is included.
