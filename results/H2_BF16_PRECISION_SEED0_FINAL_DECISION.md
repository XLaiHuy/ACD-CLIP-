# H2 BF16 Seed0 precision / seed decision

`FINAL_DECISION=PASS`  
`LAST_FULL_TRAIN=A_BF16_SEED0_E15`  
`LAST_FULL_TRAIN_VALIDITY=PASS`  
`SOURCE_AUDIT_STATUS=PASS`  
`HISTORICAL_SEED0_SOURCE_AUDIT=UNAVAILABLE`  
`GENERALIZATION_BOTTLENECK=POSSIBLE`  
`MAX_ADDITIONAL_FULL_TRAINS=1; ACTUAL_ADDITIONAL_FULL_TRAINS=1; MORE_TRAINING_AUTHORIZED=NO`  
`WAITING_FOR_USER_APPROVAL=YES`

| arm | Medical AUROC | Medical AP | MVTec AUROC | MVTec AP |
|---|---:|---:|---:|---:|
| A FP16+FP32 Seed0 | 91.2518 | 39.4684 | 90.0413 | 45.1593 |
| A BF16 Seed0 | 90.2259 | 36.3341 | 86.1169 | 40.5415 |
| A BF16 Seed1 | 88.8694 | 34.6473 | 84.8983 | 41.8373 |

BF16 Seed0 minus mixed Seed0: Medical `-1.0259/-3.1342`; MVTec `-3.9244/-4.6179` (AUROC/AP, percentage points).

`MEAN_PIXEL_AP_DELTA_BF16_SEED0_VS_MIXED_SEED0=-3.876039` and `MEAN_PIXEL_AUROC_DELTA_BF16_SEED0_VS_MIXED_SEED0=-2.475140`. The predeclared clear-loss rule applies: AP is below −2 points and both target domains decline. `PRECISION_QUALITY_DEGRADATION=LIKELY`; `RECOMMENDED_PRECISION=MIXED_FP16_FP32` because historical A itself records zero nonfinite events. BF16 Seed0 versus Seed1 is mixed across domains, so `SEED_INITIALIZATION_SENSITIVITY=POSSIBLE`, not the primary explanation.

The historical source checkpoint is unavailable, hence its source observability audit remains unavailable. Its target metrics are nevertheless stored under the exact current target evaluator contract. Seed1 source ranking/feature separation was healthy, DFG routing suspicious, and pixel localization a bottleneck; the new cross-domain Seed0 result makes precision-policy quality the primary bottleneck rather than generalization alone. No further training is authorized.
