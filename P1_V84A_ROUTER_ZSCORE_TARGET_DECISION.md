# P1-v8.4-A Router patch-zscore target decision

## Scientific isolation

The sole Router change is `patch_zscore_softmax`:

`q_router = softmax((gain_rel - mean(gain_rel)) / clamp(std(gain_rel), 1e-12))`.

Margin eligibility remains `valid & best_gain_rel > 0 & margin_rel > 0.10`.
The target helper is detached and introduces no parameter. It preserves the
Router raw winner/order; factor q/responsibility, candidate logits, ACT
g_route teacher/masks, residual/routed correction, rho, and P1-v8.3 defaults
are unchanged. A zero-spread target is finite uniform, and a focused test plus
the frozen audit prohibit it on an eligible patch.

## Frozen target audit

The sole 300-batch forward-only audit reconstructed manifest
`20575581d04b90b1221130d9870e5a6b50466d942c88eeb8bf18c39c474cc25b`, exact
support (1,071,261 overall; 1,066,751 normal; 4,510 anomaly), and anomaly
winner counts F2=2,215, F3=1,895, F4=400. The source checkpoint SHA256 was
`96f679b2e18f4e352157494f7198414b66f66024a5cc023f5ff046c39dcaa3a3`; model
hash was unchanged with zero backward/optimizer/scheduler steps.

q was finite and preserved all eligible raw winners. Its normalized-entropy
p50 was 0.60235 overall and 0.79537 anomaly, below 0.98; p95 max-q was
0.76839 overall and 0.62192 anomaly, so it was not one-hot. This accepted
`ROUTER_ZSCORE_TARGET_USABLE`.

## Calibration and smoke

Fresh-init no-step calibration found historical lambda 0.10 unsafe
(weighted p95/max 112.96166/121.03628). The sole deterministic rescale
selected `lambda_router=0.00044262806523447237`, with verified candidate
weighted p95/max 0.50000/0.53574. State hash was unchanged.

The single 8B smoke completed 8/8 batches and 2/2 steps. Support was alive,
the Router parameter delta was nonzero (L2 0.08895), gradients were finite,
and every reconstruction/MAIN/rho invariant held. It did not trigger an 8B
exit gate.

## One 300B evidence run

The single fresh run completed 300 batches and 50 steps in 546.80 seconds.
No NaN, Inf, OOM, invariant failure, gradient-scale failure, or routing
collapse occurred. Router support remained material (0.89666 cumulative;
0.89649 normal; 0.92554 anomaly). Router usage remained distributed:
`[0.25808, 0.24330, 0.24689, 0.25174]`.

Across saved non-overlapping diagnostic windows, target-winner agreement rose
from 0.31364 (batches 1-50) to 0.53159 (251-300); student entropy moved from
0.99903 to 0.99853 while targets stayed non-uniform. The final Router
parameter delta was L2 2.08311 (4.33% of fresh-init Router norm). Runtime
weighted Router/MAIN ratios were median/p95/max
0.05119/0.12760/0.16311.

Base/routed/ActualGated diagnostics are retained in the compact 300B summary.
ActualGated suppressed 64.6% of observed normal ACT=1 routed damage while
retaining 30.4% of anomaly ACT=1 benefit. These are semantic diagnostics, not
medical evaluation or a final performance claim.

## Decision and limits

`ROUTER_300B_EVIDENCE_READY_FOR_REVIEW`

No second target, replay, calibration, 8B/300B run, target-temperature change,
margin change, reweighting, capacity change, medical evaluation, or push is
authorized. The next action is scientific discussion only.
