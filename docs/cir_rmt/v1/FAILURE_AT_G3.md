# CIR_DFG_RMT_V1 — frozen failure at G3

- `arch_id`: `CIR_DFG_RMT_V1`
- terminal commit: `e1a61b71b48542ad5fa0a260883dd0a8c9374a42`
- status: `FROZEN_FAILED`
- failure gate: `G3_REAL`
- failure reason: `SIGN_CONVENTION_FAILURE`

## Evidence

G2 REAL alpha=0 parity passed on the real CLIP asset and VisA image:

- native DFG weight max error: `0`
- alpha=0 logits max error: `1.43e-6`
- alpha=0 logits mean error: `2.33e-7`
- final map max error: `1.19e-7`
- final map mean error: `8.41e-9`
- NaN/Inf: none

The bounded VisA source preflight used the deterministic first 16 manifest rows.
Positive transport degraded monotonically: pixel AUROC/AP were `0.595218/0.002530`
(alpha 0), `0.5529/0.002157` (alpha +0.25),
`0.520078/0.001838` (alpha +0.50), and `0.4737/0.001560`
(alpha +1.00).

Sign falsification gave pixel AUROC/AP `0.623325/0.002829` at alpha -0.25
and `0.636261/0.003039` at alpha -0.50. The induced anomaly-minus-normal
margin change was `-0.007886` for V1 alpha +0.50 and `+0.007696` for
alpha -0.50. Delta retained weak source relation (signed delta AUROC
`0.570933`; absolute delta AUROC `0.544486`), while shuffled +0.50 gave
pixel AUROC `0.445725` and pixel AP `0.001610`.

## Release disposition

V1 is not releaseable. Full training was correctly not launched and a V1
release lock was not generated. This is a scientific transport-direction
falsification, not an implementation-parity bug. V1 history remains unchanged.
