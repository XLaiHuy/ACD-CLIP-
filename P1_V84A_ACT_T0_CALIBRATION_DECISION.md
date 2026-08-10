# P1-v8.4-A ACT Zero-Boundary Calibration Decision

## ACT threshold and semantics

The ACT teacher object is `g_route`: the ACT=1 utility gain of the current Router-weighted residual correction. The candidate threshold is changed only for P1-v8.4-A from `0.02` to `0.0`; the legacy `h6_utility_gain_threshold=0.02` and P1-v8.3 behavior remain unchanged.

The semantic reason is causal: `g_route <= 0` means the correction ACT would apply is non-beneficial or harmful, while `g_route > 0` means it is beneficial. At threshold zero, the ambiguous interval is empty. No positive `T_off` or fabricated OFF examples were introduced.

## Frozen support consequence

Using the committed teacher-semantics audit only, with no new replay:

| region | ON (`g_route > 0`) | OFF (`g_route <= 0`) |
|---|---:|---:|
| overall | 1.057245% | 98.942757% |
| normal | 0.475017% | 99.524981% |
| anomaly | 100.000000% | 0.000000% |

The natural boundary produces no ambiguous support. Label imbalance is reported as evidence, not treated as invalid; no loss rebalance was applied.

## Fresh-init no-step calibration

The established calibration methodology ran 24 six-microbatch windows (144 VisA/train seed-0 microbatches) with fresh OpenAI CLIP-only initialization, image 518, batch 1, accumulation 6, FP32, AMP off, TF32 off, gradient checkpointing, and P1-v8.4-A. The protocol checkpoint was used only for configuration; no checkpoint weights were loaded.

- Old `lambda_act`: `5.270823562063741e-05`
- Recalibrated `lambda_act`: `7.435420936678605e-05`
- ACT labels: 312,870 ON (60.298573%), 205,998 OFF (39.701427%), 0 ambiguous, 518,868 valid group-patches
- Physical patch context: 172,192 normal and 764 anomaly patches; Router informative support 0 under its locked threshold
- Raw ACT/main ratios: median `4901.82421875`, p95 `6724.56884765625`, max `6934.45068359375`
- Weighted ACT/main ratios: median `0.3644712865`, p95 `0.5`, max `0.5156056285`
- Safety contract: raw finite; weighted max <= 1; weighted p95 <= 0.5
- Integrity: parameter hash before/after identical (`50a7c4b1160561ae3ab934f34895311b50882706c1acb4e9f6349231ad44fbc6`), gradients clear, zero optimizer steps, zero scheduler steps, no parameter mutation

## Decision

`ACT_THRESHOLD_AND_LAMBDA_READY`

## Next discussion

Discuss an ACT-only fresh 8B experiment using the zero utility boundary and recalibrated lambda. Do not launch it automatically.

## Forbidden actions

No 8B or 300B training in this task, no new replay, no Router change, no margin gating, no loss reweighting, no capacity change, no positive `T_off`, and no automatic threshold selection from desired support counts.
