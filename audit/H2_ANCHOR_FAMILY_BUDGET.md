# Final H2 Anchor family-budget audit

This is the final source-only decision record for the candidate H2/CIR-V2 checkout.

## Scope and fixed decision

- Historical audit: frozen H2 E1/E5/E10/E15 model-only checkpoints, one deterministic augmented VisA train batch, geometry only. It makes no optimizer, Adam, RNG, or update-parity claim.
- Bounded mechanism audit: a fresh shared E1 followed by native H_short and family-safe A_safe_short through E3, with the same full-state resume, loader identity, augmentation hashes, and fixed optimizer/scheduler/scaler state.
- No Medical, MVTec, target-label, CIR-training, full-training, lambda sweep, or rho sweep was used.
- Global Anchor formula remains `sum_i ||theta_i-theta_ref_i||^2 / (sum_i ||theta_ref_i||^2 + eps)`.
- `ANCHOR_LAMBDA=0.001` is unchanged. `ANCHOR_FAMILY_BUDGET_RHO=0.10` is fixed.
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

Fresh root: `/tmp/h2_anchor_family_short_20260902_retry2`.

- H_short and A_safe_short both passed E2/E3 with two batches per epoch.
- Batch identities and transformed image/mask hashes matched exactly across H/A.
- Non-finite loss/gradient skips were zero.
- H_short stayed native H2 with zero effective Anchor gradient.
- A_safe_short stayed within the family cap; maximum effective active-family ratio was `0.0016128024402913622`.
- The effective global Anchor contribution was nonzero after drift, but remained below the report's `1e-6` active threshold (`8.507442859846122e-7` maximum), so the result is `FAMILY_SAFE_BUT_NEGLIGIBLE`.
- No `40,000x` effective-ratio pathology occurred.
- H/A differences were introduced by the image-only Anchor branch; later text-side state deltas are downstream consequences of subsequent task updates from the changed image state.

## Gate outcome

- `UNSAFE_GLOBAL_SCALING_DISABLED=PASS`
- `E5_FAMILY_AUDIT=PASS`, `E10_FAMILY_AUDIT=PASS`, `E15_FAMILY_AUDIT=PASS`
- `TASK_NEAR_ZERO_RULE=PASS`
- `FAMILY_PARTITION_COMPLETE=PASS`
- `H_SHORT=PASS`, `A_SAFE_SHORT=PASS`, `H_A_ONLY_EXPECTED_DIFFERENCE=PASS`
- `ANCHOR_STATUS=FAMILY_SAFE_BUT_NEGLIGIBLE`
- `H_FULL_TRAIN_READY=YES`, `A_FULL_TRAIN_READY=NO`, `C_FULL_TRAIN_READY=YES`, `AC_FULL_TRAIN_READY=NO`
- `CIR_IMPLEMENTATION=EXACT_V2`; `CIR_CHANGED=NO`
- `MEDICAL_USED=NO`, `MVTEC_USED=NO`, `FULL_TRAIN_RUN=NO`, `PERFORMANCE_WINNER_DECIDED=NO`
- Full repository tests: `43 passed, 0 failed`.

The single scientific fix is the opt-in family-safe Anchor gradient budget. No target or CIR implementation was changed.

## Machine-readable gate fields

UNSAFE_GLOBAL_SCALING_DISABLED=PASS
ANCHOR_FORMULA=sum_i||theta_i-theta_ref_i||^2/(sum_i||theta_ref_i||^2+eps)
ANCHOR_FAMILY_BUDGET_RHO=0.10
ANCHOR_LAMBDA=0.001
ANCHOR_LAMBDA_CHANGED=NO
E5_FAMILY_AUDIT=PASS
E10_FAMILY_AUDIT=PASS
E15_FAMILY_AUDIT=PASS
TASK_NEAR_ZERO_RULE=PASS
FAMILY_PARTITION_COMPLETE=PASS
MAX_EFFECTIVE_ACTIVE_FAMILY_RATIO=0.0016128024402913622
H_SHORT=PASS
A_SAFE_SHORT=PASS
H_A_ONLY_EXPECTED_DIFFERENCE=PASS
ANCHOR_STATUS=FAMILY_SAFE_BUT_NEGLIGIBLE
H_FULL_TRAIN_READY=YES
A_FULL_TRAIN_READY=NO
C_FULL_TRAIN_READY=YES
AC_FULL_TRAIN_READY=NO
CIR_IMPLEMENTATION=EXACT_V2
CIR_CHANGED=NO
MEDICAL_USED=NO
MVTEC_USED=NO
FULL_TRAIN_RUN=NO
PERFORMANCE_WINNER_DECIDED=NO
TESTS=43/0
