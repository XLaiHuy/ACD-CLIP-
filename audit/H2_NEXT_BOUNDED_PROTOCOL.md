# H2 next bounded protocol: fixed reliability-residual stage fusion

`PROTOCOL_STATUS=PREREGISTERED_NOT_EXECUTED`

`USER_APPROVAL_REQUIRED=YES`

`PROTOCOL_ID=H2_FIXED_E1_RELIABILITY_RESIDUAL_STAGE_FUSION_R1`

`SOURCE=VisA_ONLY`

`TARGET_INFERENCE=NO`

`TARGET_TUNING=NO`

`MEDICAL=NO`

`MVTEC=NO`

`FULL_E15=NO`

`HYPERPARAMETER_SWEEP=NO`

`MAX_ATTEMPTS_PER_ARM=500`

No runner, optimizer, scaler, or training process may be started from this
protocol until the user approves it.

## Scientific question

Does reducing the contribution of harmful late-stage maps while retaining the
stage-1 signal preserve source anomaly response better than equal fusion,
without relying on an increase in E1 feature cosine?

The mechanism is downstream and deliberately bounded. It does not claim to
repair every late-stage parameter drift; it tests whether late-stage
overadaptation is harmful because its output contribution is too large.

## Matched arms

### Control

`A_FUSE_SHORT_R1_CONTROL`

Start from the shared E1 full-state checkpoint and run the existing H2 Safe
Anchor configuration with the historical equal stage fusion
`[1/3, 1/3, 1/3]`.

### Candidate

`A_FUSE_SHORT_R1_CANDIDATE`

Start from the exact same shared E1 full-state checkpoint and run the same H2
configuration, changing only the fixed stage fusion weights to
`[0.3736138197153701, 0.3270300383596602, 0.2993561419249697]`.

### Only scientific difference

The candidate uses the fixed convex stage mixture above wherever the native
three stage segmentation maps are fused during the bounded run and endpoint
evaluation. The control uses equal fusion. Everything else is held identical:

* Safe Anchor remains enabled with the existing `anchor_lambda`
  `0.0021633926715180626`, family budget `rho=0.10`, and E1 reference;
* no functional feature-anchor term is present in either arm;
* no CIR change, DFG change, SS2D change, prompt change, loss change, LR
  change, optimizer change, horizon change, or parameter-family mask is added;
* stage-specific maps remain available for diagnostics; the candidate does not
  delete or detach stages 2 or 3;
* no new trainable or inference parameter is introduced.

This is the fixed, non-learned simplification of the previously frozen
`BOUNDED_RELIABILITY_RESIDUAL_STAGE_FUSION` candidate. It is not a learned gate
and it is not a sweep.

## Start state and execution identity

`START_CHECKPOINT=runs/h2_clean_factorial_e20_20260902_ampfix/shared_e1/adapter_1.pth`

`START_CHECKPOINT_SHA256=7f9176b7ef53b572935567c574535075a573175b2aa83505d043a71d45b12b35`

The checkpoint must be verified before either arm starts. It is the shared E1
full-state checkpoint, not either R1 endpoint. The expected source identity is:

| field | fixed value |
|---|---|
| seed | `0` |
| model | `ViT-L-14-336` |
| image size | `518` |
| n groups | `3` |
| source dataset | `VisA` |
| precision | historical mixed `fp16` autocast |
| parameter/optimizer storage | `fp32` |
| GradScaler | enabled |
| deterministic algorithms | enabled |
| DFG | attention, tau `8.0`, SS2D weight residual |
| Safe Anchor | enabled, family budget `0.10` |
| attempts | exactly `500` attempted per arm |

The existing R1 16-batch manifest is reused for preflight and the same
deterministic augmentation/file order is required. The candidate and control
must have identical file names, labels, transformed-image hashes, and mask
hashes at every attempted batch.

## Source-only coefficient calibration

The candidate weights are fixed before training by one deterministic rule; no
coefficient is selected from a candidate endpoint and no target score is
observed.

1. Use the already frozen 96-image VisA source evaluation subset: first four
   normal and first four anomalous manifest-order images in each VisA category.
2. Use the shared E1 checkpoint and compute the three stage AP values already
   recorded in `audit/H2_FUNC_ANCHOR_R1_SOURCE_EVAL.json`:

   ```text
   AP_E1 = [0.25648650726844374,
            0.22450666405925707,
            0.20550848823040294]
   ```

3. Set `w_i = AP_E1_i / sum_j(AP_E1_j)`, with no smoothing, clipping, or
   search. The resulting fixed weights are:

   ```text
   w = [0.3736138197153701,
        0.3270300383596602,
        0.2993561419249697]
   ```

The equal-fusion control uses `[1/3, 1/3, 1/3]`; it does not use the calibrated
weights. This rule is source-only, deterministic, and frozen before either
bounded arm is run.

## Bounded execution and numerical validity

Run the two arms from the same E1 checkpoint with the same seed and the same
500-attempt E2/E3 source schedule used by R1. Run a 16-batch parity preflight
first. The bounded result is numerically valid only when all of the following
hold for both arms:

* no non-finite loss, trainable parameter, or optimizer-state failure;
* zero or one isolated recoverable non-finite-gradient skip per arm;
* no consecutive skip and no unequal successful-step count caused by arm
  divergence;
* exact augmented-batch identity across arms for all 500 attempts;
* finite endpoint checkpoint and complete run manifest.

If either arm fails this numerical contract, the result is `INVALID_NUMERICAL`
and cannot support a mechanism pass. Do not repair it by changing precision,
seed, lambda, or horizon in this protocol.

## Fixed endpoint evaluation

Evaluate both endpoints on the same 96-image clean VisA subset used for the
coefficient rule. Do not evaluate Medical or MVTec. Record:

* final pixel AUROC and AP;
* stage-1, stage-2, and stage-3 pixel AUROC and AP;
* positive, negative, boundary, interior, near-background, and far-background
  mean, median, and p99 score coverage;
* stage-1 health separately from stage-2/3 ranking;
* stage-1/2/3 feature cosine to E1 and CKA/drift as descriptive geometry
  diagnostics only;
* parameter-family drift and update telemetry using the existing family
  partition; no family is newly masked;
* numerical validity, successful steps, and exact batch parity.

## Mechanism metric and pass/fail gates

The direct mechanism metric is not E1 cosine. Before softmax, record each
stage's contribution to the fused abnormal logit and compute the fixed
late-stage contribution ratio

```text
R_late = mean_over_images(
    (w2 * mean_abs(logit_stage2) + w3 * mean_abs(logit_stage3)) /
    (w1 * mean_abs(logit_stage1) + w2 * mean_abs(logit_stage2)
       + w3 * mean_abs(logit_stage3) + 1e-12)
)
```

Use the corresponding control weights for the control metric. A mechanism
screen `PASS` requires all gates below; otherwise it is `FAIL`:

1. **Numerical validity:** both arms pass the bounded numerical contract.
2. **Mechanism movement:** candidate `R_late` is lower than control and the
   logged per-stage contribution change agrees with the preregistered weight
   direction. A rise in E1 cosine alone is not evidence of movement.
3. **Anomaly preservation:** candidate final AP and AUROC are each no lower
   than control beyond `1e-6`, and both positive and interior coverage mean
   and median are each no lower than control beyond `1e-6`.
4. **Stage-1 health:** candidate stage-1 AP and AUROC are each no lower than
   control beyond `1e-6`.
5. **Late-stage ranking:** stage-2 and stage-3 AP/AUROC are reported. If both
   late stages lose AP and the final coverage gate is not met, classify the
   mechanism as a failure rather than calling the fusion change a success.
6. **No geometry-only pass:** a candidate that only increases stage-2/3 E1
   cosine fails unless gates 1--4 also pass.

No target metric, published comparator, or target-side qualitative result may
be used for selection, calibration, or interpretation of this screen.

## Stop conditions

Stop immediately after the two bounded endpoints and source evaluation. Do not
start a full E15/E20 run, run a seed search, retry the functional anchor,
change lambda, change horizon, run Medical/MVTec inference, or add another
candidate. The next decision must be based only on this protocol's source and
validity artifacts.
