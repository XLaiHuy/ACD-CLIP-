# P1-v8.4-A Router z-score target: 8-batch decision

Source state: `18b315f6dbb1430d24cb1e7b5c9549b2ea4b28a2`.

The one frozen target audit reconstructed the fingerprinted baseline exactly
(manifest SHA256 `20575581d04b90b1221130d9870e5a6b50466d942c88eeb8bf18c39c474cc25b`):
selected support was 1,071,261 overall, 1,066,751 normal, and 4,510 anomaly,
with anomaly winners F2=2,215, F3=1,895, and F4=400. The selected target is
only `patch_zscore_softmax`: `softmax((gain - mean(gain)) / clamp(std(gain),
1e-12))`. It retained raw-winner agreement and produced median normalized
target entropy 0.60235 overall and 0.79537 anomaly (both below 0.98).

Fresh-init no-step calibration (24 six-microbatch windows, seed 0,
VisA/train, FP32, AMP/TF32 off) selected
`lambda_router=0.00044262806523447237`. Router raw gradient ratios were
median/p95/max = 981.95490/1129.61658/1210.36279; weighted ratios were
0.43464/0.50000/0.53574. Its model-state hash was unchanged and optimizer
and scheduler steps were both zero.

The single fresh 8-batch smoke used that value with margin eligibility
`valid & best_gain > 0 & margin_rel > 0.10`, fixed rho=0.05, the unchanged
factor and ACT controls, and `patch_zscore_softmax`. It completed 8/8
microbatches and two optimizer steps; all finite, residual and surgery
reconstructions were exact, MAIN exact-change was zero, and rho remained
non-trainable. Router gradients were alive (final norm `4.4787e-05`),
router utility loss was finite, and supervised router support was present at
every recorded state (final cumulative fraction 0.91522). The teacher target
was non-uniform (final cumulative normalized entropy 0.73832); the early
student router remained near-uniform as expected after two updates
(entropy 0.99902), without a collapse or invariant failure.

Decision: `ROUTER_TARGET_8B_READY_FOR_300B`.

The only authorized next experiment is one fresh 300-microbatch / 50-step
run with this exact target and lambda. Do not change Router eligibility,
target temperature/formula, ACT, factor teacher, loss weights, capacity,
or rho; do not run medical evaluation or a second 300-batch attempt.
