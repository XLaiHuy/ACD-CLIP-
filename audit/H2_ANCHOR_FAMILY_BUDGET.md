# H2 Anchor family-budget audit and E20 activation decision

This is the final source-only decision record for the candidate H2/CIR-V2 checkout.

## Scope and fixed decision

- Historical audit: frozen H2 E1/E5/E10/E15 model-only checkpoints, one deterministic augmented VisA train batch, geometry only. It makes no optimizer, Adam, RNG, or update-parity claim.
- Bounded mechanism audit: a fresh shared E1 followed by native H_short and calibrated A_active_short through E5, with the same full-state resume, loader identity, augmentation hashes, and fixed optimizer/scheduler/scaler state.
- No Medical, MVTec, target-label, CIR-training, full-training, lambda sweep, or rho sweep was used.
- Global Anchor formula remains `sum_i ||theta_i-theta_ref_i||^2 / (sum_i ||theta_ref_i||^2 + eps)`.
- The historical lambda was `0.001`; the one frozen active calibration is `R_MED=23.11184680352771`, target effective ratio `0.05`, and `ANCHOR_LAMBDA=0.0021633926715180626`. `ANCHOR_FAMILY_BUDGET_RHO=0.10` is fixed.
- `--anchor_gradient_budget` is opt-in and disabled by default. A/AC launchers enable it with rho `0.10`; H/C remain native.
- The task floor is `max(1e-12, 1e-6 * global_task_grad_norm / sqrt(total_trainable_parameters))`; every audited step records it.

The historical global `0.1 / median(E5,E10,E15 raw global ratio)` rule is retained only as provenance. It is marked `DEPRECATED_UNSAFE_GLOBAL_SCALING`, with `GLOBAL_LAMBDA_UPSCALING_AUTHORIZED=NO`, and is not executed.

## Complete image-adapter partition

| Family | Parameter count |
| --- | ---: |
| `lora_adapters` | 6,514,176 |
| `m_i_w` | 3,072 |
| `seg_proj` | 1,379,328 |
| `det_proj` | 1,379,328 |
| `seg_layer_norms` | 4,608 |
| `det_layer_norms` | 4,608 |
| `vision_text_q` | 589,824 |
| `vision_text_k` | 589,824 |
| `dfg_ss2d_branches` | 3,550,464 |
| `dfg_raw_gamma` | 3 |
| `direction_logits` | 12 |
| `remaining_image_adapter_params` | 0 |

The partition is complete and disjoint. The JSON audit reports, for every family at E1/E5/E10/E15, reference norm, displacement norm, task norm, raw Anchor norm, raw and lambda-scaled ratios, task/Anchor cosine, floor, finite status, and classification.

## Historical family geometry

| Checkpoint | Largest raw family ratio | Lambda times raw/task | Dominant or active evidence | Global raw ratio | Global lambda ratio |
| --- | ---: | ---: | --- | ---: | ---: |
| E5 | 1,130,628.5269823656 (`dfg_ss2d_branches`) | 1,130.6285269823657 | DFG branch `ANCHOR_DOMINANT`; vision-text K task-active; vision-text Q moderate | 0.0038510432618746628 | 3.8510432618746625e-6 |
| E10 | 819,818.2472821391 (`dfg_ss2d_branches`) | 819.8182472821392 | DFG branch `ANCHOR_DOMINANT`; vision-text K task-active; vision-text Q moderate | 0.02249237331791408 | 2.2492373317914082e-5 |
| E15 | 12,718,060.14226764 (`dfg_ss2d_branches`) | 12,718.060142267641 | DFG branch `ANCHOR_DOMINANT`; vision-text K task-active; vision-text Q moderate | 0.018389481745311707 | 1.838948174531171e-5 |

DFG raw gamma and direction logits are classified `TASK_NEAR_ZERO` in these historical probes; ratios are not used to infer activity for those families. The result is the reason global upscaling is unsafe: it masks family imbalance while amplifying a dormant/near-zero-task DFG family.

## Fresh bounded H/A mechanism result

Fresh root: `/tmp/h2_anchor_family_short_e20_20260902`.

- H_short and A_active_short both passed E2-E5 with two batches per epoch.
- Batch identities and transformed image/mask hashes matched exactly across H/A.
- Non-finite loss/gradient skips were zero.
- H_short stayed native H2 with zero effective Anchor gradient.
- A_active_short stayed within the family cap; maximum effective active-family ratio was `0.09999937754084845`.
- A meaningful `TASK_ACTIVE` family reached effective ratio `0.03957792788580696`; the post-drift global effective contribution was nonzero.
- No `40,000x` effective-ratio pathology occurred; maximum raw family ratio was `50.81989419188151`.
- H/A differences were introduced by the image-only Anchor branch; later text-side state deltas are downstream consequences of subsequent task updates from the changed image state.

## Gate outcome

- `UNSAFE_GLOBAL_SCALING_DISABLED=PASS`
- `HISTORICAL_E5_FAMILY_AUDIT=PASS`, `HISTORICAL_E10_FAMILY_AUDIT=PASS`, `HISTORICAL_E15_FAMILY_AUDIT=PASS`
- `TASK_NEAR_ZERO_RULE=PASS`
- `FAMILY_PARTITION_COMPLETE=PASS`
- `H_SHORT=PASS`, `A_ACTIVE_SHORT=PASS`, `H_A_ONLY_EXPECTED_DIFFERENCE=PASS`
- `ANCHOR_STATUS=FAMILY_SAFE_ACTIVE`, `ACTIVATION_GATE=PASS`
- `H_FULL_TRAIN_READY=YES`, `A_FULL_TRAIN_READY=YES`, `C_FULL_TRAIN_READY=YES`, `AC_FULL_TRAIN_READY=YES`
- `CIR_IMPLEMENTATION=EXACT_V2`; `CIR_CHANGED=NO`
- `MEDICAL_USED=NO`, `MVTEC_USED=NO`, `FULL_TRAIN_RUN=NO`, `PERFORMANCE_WINNER_DECIDED=NO`
- Full repository tests: `46 passed, 0 failed`.

The single scientific fix is the opt-in family-safe Anchor gradient budget. No target or CIR implementation was changed.

## Machine-readable gate fields

UNSAFE_GLOBAL_SCALING_DISABLED=PASS
ANCHOR_FORMULA=sum_i||theta_i-theta_ref_i||^2/(sum_i||theta_ref_i||^2+eps)
ANCHOR_FAMILY_BUDGET_RHO=0.10
ANCHOR_LAMBDA=0.0021633926715180626
ANCHOR_LAMBDA_OLD=0.001
ANCHOR_LAMBDA_CHANGED=YES
ANCHOR_R_MED=23.11184680352771
ANCHOR_TARGET_EFFECTIVE_RATIO=0.05
E5_FAMILY_AUDIT=PASS
E10_FAMILY_AUDIT=PASS
E15_FAMILY_AUDIT=PASS
TASK_NEAR_ZERO_RULE=PASS
FAMILY_PARTITION_COMPLETE=PASS
MAX_EFFECTIVE_ACTIVE_FAMILY_RATIO=0.09999937754084845
H_SHORT=PASS
A_ACTIVE_SHORT=PASS
H_A_ONLY_EXPECTED_DIFFERENCE=PASS
MEANINGFUL_TASK_ACTIVE_RATIO_MAX=0.03957792788580696
ANCHOR_ACTIVATION_GATE=PASS
ANCHOR_STATUS=FAMILY_SAFE_ACTIVE
H_FULL_TRAIN_READY=YES
A_FULL_TRAIN_READY=YES
C_FULL_TRAIN_READY=YES
AC_FULL_TRAIN_READY=YES
CIR_IMPLEMENTATION=EXACT_V2
CIR_CHANGED=NO
MEDICAL_USED=NO
MVTEC_USED=NO
FULL_TRAIN_RUN=NO
PERFORMANCE_WINNER_DECIDED=NO
TESTS=46/0
