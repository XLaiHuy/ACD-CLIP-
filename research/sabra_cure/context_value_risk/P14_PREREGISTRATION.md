# SABRA-CURE Contextual Value-Risk V1 (P14)

Status: `FROZEN_BEFORE_IMPLEMENTATION_AND_RESULTS`.
Parent: `ed867055a2459ccfcf1fccb12d5e8990a8a3117e`.

P14 is one source-side, 12-fold VisA LOCO confirmatory study.  It preserves all
R2-v2 patch-level direction and 22-feature harm-risk definitions, models,
scalers, nested exclusions, proposal signs, correction operator, and fixed
alpha `.25`.  No MVTec/Medical read, CLIP forward, Phase2B training, new
representation, patch benefit target, alpha/coverage/threshold sweep, R3, or
R4 is permitted.

The only new component is an image-context ridge controller.  Harm risk orders
proposals: SAFE20 accepts `risk <= tau20`, EXPAND40 accepts `risk <= tau40`,
where both thresholds are leakage-safe OOF quantiles `.20` and `.40` using
`numpy.quantile(..., method='linear')` with deterministic `<=`.  Every image
uses one complete patch policy; the controller never selects individual patches.

The source label is the exact marginal class-pAP counterfactual
`V_j = AP(class: image j E40, all others S20) - AP(class: all S20)`.  It is
image-level, GT-bearing only during source training, non-additive, and includes
normal images.  No per-patch AP attribution is made.  Exactly one P14 attempt
is allowed; all terminal states publish and stop for user review.
