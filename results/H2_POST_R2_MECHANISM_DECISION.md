# H2 Post-R2 Mechanism Decision

`PROTOCOL_ID=H2_POST_R2_CAUSAL_DECOMPOSITION_2X2_FROZEN_CROSS_EVALUATION`

`ENDPOINT_IDENTITY=PASS`

`CROSS_EVAL_REPRODUCTION=PASS`

`DOMINANT_PATTERN=INFERENCE_ONLY_FUSION_PROMISING`

`BOTTLENECK_STATUS=REFINES`

`REFINED_BOTTLENECK=LATE_STAGE_TRAINING_OVERADAPTATION_WITH_ANOMALY_COVERAGE_VS_HARD_BACKGROUND_SEPARATION_TRADEOFF`

## Evidence

- C_W minus C_EQ final AP: `0.001450442814`; final AUROC: `0.004350496932`.
- T_EQ minus C_EQ final AP: `-0.006241956666`.
- Training-stage AP deltas: stage 1 `-0.012786217261`, stage 2 `-0.017633093426`, stage 3 `0.012361037277`.
- T_EQ minus C_EQ near-background p99: `0.046235395670`.
- Near-background positive/interior pairwise inversion deltas: `0.004535000000` / `0.002945000000`.

The frozen cross-evaluation supports weighted fusion as an inference-only
candidate: C_W improves both final AP and AUROC at the control-trained
checkpoint, while weighted training causes the AP and stage-1/2 degradation.
The near-background tail and rank inversions rise under weighted training,
refining the prior diagnosis to a coverage-versus-hard-background tradeoff.

## Ranked next mechanisms

1. **Inference-only fixed stage fusion** — selected; strong direct evidence,
   no inference overhead beyond the weighted sum, and no training run here.
2. **Normal/background-selective E1 preservation** — conceptual only; do not
   implement or assign coefficients without a new preregistration and approval.
3. **Family/stage-scoped gradient budget or conflict control** — lower priority;
   requires separate evidence and approval.

`SELECTED_NEXT_MECHANISM=inference-only fixed stage fusion`

`NEXT_BOUNDED_EXPERIMENT_JUSTIFIED=YES`

No new training, optimizer step, target inference, Medical/MVTec inference,
full E15, tuning, or sweep was performed. The original R2 FAIL decision is not
changed.
