# P1-v8.4-A routed-ACT 300B decision

## Run and isolation

Exactly one fresh run was executed from source commit `9c68be4cd9e03dfc898ce20ace2f0e6ceb595978`.
It used OpenAI CLIP-only initialization (no Phase2B checkpoint), seed 0, VisA/train, image 518,
batch 1, accumulation 6, FP32, AMP off, TF32 off, and gradient checkpointing.  The run executed
300 microbatches and 50 optimizer steps, with milestones at 50/100/150/200/250/300.  The exact
launch configuration is retained in `config.json` and the compact command contract was:

```text
ACDCLIP_DATA_ROOT=/workspace/data/med_visa/data \
ACDCLIP_CLIP_VITL14_336=/workspace/ACD-CLIP-p1v84a/model/ViT-L-14-336px.pt \
/workspace/.venv-p1v84a/bin/python train.py --dataset VisA --img_size 518 --epoch 1 \
  --save_path runs/p1_v84a_gpu/act_routed_t0_300b_seed0_attempt1 --seed 0 \
  --batch_size 1 --grad_accum_steps 6 --precision fp32 --grad_checkpointing \
  --h6_progress 1 --h6_progress_version P1-v8.4-A \
  --h6_act_gain_threshold 0.0 --lambda_h6_act 7.435420936678605e-05 \
  --lambda_h6_factor 0.03 --lambda_h6_router 0.10 \
  --h6_factor_tau_utility 0.05 --h6_router_tau_utility 0.05 \
  --h6_router_gain_threshold 0.02 --h6_utility_entropy_threshold 0.98 \
  --h6_primary_anchored_factor_surgery --h6_router_support_normalized \
  --h6_trajectory_milestones 50 100 150 200 250 300 --h6_smoke_max_batches 300
```

The tested variables were only the approved routed teacher, zero utility boundary, and calibrated
`lambda_act`.  Router formulation, factor responsibility, losses, capacities, optimizer, surgery,
and `rho=0.05` were unchanged.  No Router margin gate, threshold search, loss rebalance, capacity
change, medical evaluation, or second attempt was performed.

Historical context: the preceding ACT configuration used the old teacher object and `0.02` ACT
threshold, and its frozen evidence had no positive ACT support.  This artifact is a distinct
routed-T0 run and does not overwrite the historical 8B evidence.

## ACT support and learning

The zero-boundary labels were exact: ambiguous support was 0 in every cumulative block.  Cumulative
ON/OFF/ambiguous percentages (overall; normal; anomaly) were:

| batches | overall | normal | anomaly |
|---|---|---|---|
| 1–50 | 8.5402 / 91.4598 / 0 | 7.6928 / 92.3072 / 0 | 99.8236 / 0.1764 / 0 |
| 1–100 | 4.6388 / 95.3612 / 0 | 4.1276 / 95.8724 / 0 | 99.8442 / 0.1558 / 0 |
| 1–150 | 3.4473 / 96.5527 / 0 | 2.9640 / 97.0360 / 0 | 99.8886 / 0.1114 / 0 |
| 1–200 | 2.9408 / 97.0592 / 0 | 2.3534 / 97.6466 / 0 | 99.9306 / 0.0693 / 0 |
| 1–250 | 2.6477 / 97.3523 / 0 | 2.0140 / 97.9860 / 0 | 99.9481 / 0.0519 / 0 |
| 1–300 | 2.3173 / 97.6827 / 0 | 1.7431 / 98.2569 / 0 | 99.9521 / 0.0479 / 0 |

Each cell is ON / OFF / ambiguous percent.  Both classes occur over the full run; isolated batches
with no ON support were batch-local only.  No class balancing or fabricated anomaly-OFF labels were
used.

The required non-overlapping blocks were: 1–50 `8.5402 / 91.4598 / 0`, 51–100
`0.5317 / 99.4683 / 0`, 101–150 `1.0555 / 98.9445 / 0`, 151–200 `1.4067 / 98.5933 / 0`,
201–250 `1.4423 / 98.5577 / 0`, and 251–300 `0.6501 / 99.3499 / 0` overall.  Their normal
ON percentages were `7.6928, 0.4041, 0.6306, 0.4962, 0.6147, 0.3817`; anomaly ON percentages
were `99.8236, 100, 100, 100, 100, 100`.  Ambiguous remained zero in every block.
The corresponding ACT ON-minus-OFF separations were `+0.021829, +0.003893, -0.002053,
+0.002872, +0.003964, +0.012847`; AUROC was `0.770049, 0.548314, 0.480456, 0.652602,
0.722365, 0.843930`.  The one negative middle-block separation was not persistent and the
cumulative endpoint remained positive.

Cumulative ACT probability on teacher-ON versus teacher-OFF patches was `0.416576` versus `0.320425`
(separation `+0.096150`, AUROC `0.773609`).  The final 251–300 window was `0.274133` versus
`0.261286` (separation `+0.012847`, AUROC `0.843930`).  Cumulative normal/anomaly probability means
were `0.322661` / `0.321399`; final-window means were `0.261334` / `0.274723`.  Cumulative ACT
logit mean/std was `-0.761498 / 0.356374`; probability mean/std/min/max was `0.322653 / 0.080670 /
0.249375 / 0.513229` (final-window mean/std/min/max `0.261370 / 0.008973 / 0.249375 / 0.312911`).
The teacher-ON/teacher-OFF cross-tab at the cumulative endpoint (diagnostic probability split at
0.5 only) was ON-high 13,571, ON-low 11,233, OFF-high 27,892, OFF-low 1,017,680.  The cumulative
`g_route`/probability correlation was `-0.110128` overall (`-0.110403` normal, `-0.315066` anomaly);
the final-window correlation was `-0.373701` overall.

The ACT head moved from probability exactly 0.5 at initialization: output weight norm was 0 before
step 1 and `0.0027644` after it; the first post-update upstream ACT-feature gradient was
`0.000560640`.  The final output weight norm was `0.0408279`, and the final head raw gradient norm
was `0.000748846`.

## Utility and selective gating

Final cumulative loss-space diagnostics (lower is better) were:

| region | Base | ResidualBestSingle | ResidualOracleMulti | FullSoftRouted_ACT1 | ActualGated | HardRouted_ACT1 | Uniform_ACT1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 0.07678985 | 0.07675356 | 0.07654729 | 0.07687948 | 0.07682252 | 0.07681991 | 0.07688113 |
| normal | 0.05827778 | 0.05822251 | 0.05809626 | 0.05839091 | 0.05831760 | 0.05834191 | 0.05839185 |
| anomaly | 3.22460365 | 3.21511769 | 3.21398544 | 3.22069597 | 3.22342062 | 3.21884108 | 3.22082114 |

FullSoft ACT1 harmed normal by `0.000113130` over Base; ActualGated reduced that harm by
`0.000073314` (64.81% suppression).  FullSoft ACT1 improved anomaly by `0.003907681`; ActualGated
retained `0.001183033` (30.27% of that benefit).  Thus gating was selective rather than an
indiscriminate ACT1 suppression, while the 300-batch result is not treated as a performance claim.

## Router and safety contract

Router supervised support remained exactly zero at the canonical Router gate (`0.02`), so no Router
formulation change was made.  Final epoch dense usage was
`[[0.251869,0.245776,0.252748,0.249607],[0.253043,0.247434,0.252789,0.246734],
[0.258453,0.240124,0.247020,0.254403]]`; normalized dense entropy was
`[0.999045,0.998888,0.998645]`.  Teacher winner shares were
`[0.610678,0.059814,0.029224,0.300284]`.

All 300 batch records and 50 optimizer-step records were finite.  Residual-definition, routed
correction, ActualGated, surgery, and MAIN exact-change maxima were all exactly 0; `rho` remained
`0.05` and non-trainable.  The consistent runtime gradient proxy was ACT-head total raw gradient
norm divided by primary-anchored shared MAIN gradient norm; weighted values multiply by
`lambda_act`.  Raw ratio median/p95/max was `0.31634385 / 9.11925719 / 13.05997881`; weighted
ratio median/p95/max was `0.000251268 / 0.000678055 / 0.000971064`, within the weighted safety
limits (`p95 <= 0.5`, `max <= 1`).

Focused validation before launch passed: 62 tests, `py_compile`, and `git diff --check`.

## Decision and next authorization

The corrected routed ACT path is mechanically clean, observes both natural label classes, learns
positive ON/OFF separation, and attenuates harmful normal routed correction while retaining anomaly
benefit.  Decision: **`ACT_300B_SEMANTICALLY_HEALTHY`**.

The only next action is discussion of whether to authorize a later ACT-only experiment.  This run
does not authorize Router training, threshold search, loss reweighting, capacity changes, medical
evaluation, or another 300-batch attempt.  The generated adapter was moved outside the repository
and no model weights are committed.
