# FACTOR_ROLE_DECOMPOSITION_SCIENTIFIC_DIFF

## Root cause

`FACTOR_ROLE_DECOMPOSITION_WEAK`: the factor-state compatibility representation is identifiable but does not encode the intended normal/anomaly responsibility stably under the authorized balanced region loss.

## Decisive evidence

- Cache-only TRAIN audit: q0-q1 mean=0.4721, std=0.1403 (non-degenerate); unsupervised Normal p0=0.6125, but Anomaly p1=0.3526 / role1 recall=0.0.
- Cache-only anomaly ACT=1 rho=.05 gain=-0.008224, 95% CI [-0.009670,-0.007893].
- Real FP32 zero-step passed: finite forward/backward, zero optimizer state, exact residual/routed correction, rho=.05; intrinsic factor-responsibility gradient norm=0.005219.
- Fresh production-order short64 (11 optimizer updates; both regions supported) failed: fixed-probe Anomaly p1 0.5113 -> 0.3185 and region role1 agreement 0.8333 -> 0.0000; Normal ACT=1 rho gain +0.000233 -> -0.000751.

## Tested solution

Case B only: q_m=logsumexp_s(<patch,F[m,s]>), balanced physical N/A CE on q with lambda_h6_router=0.00044262806523447237, p_route=stopgrad(softmax(q)), and correction=.05*ACT*sum_m(p_route[m]*delta_m). Responsibility gradients were limited to factor deviations; hard/shared centers were detached. Independent Router Q/K/head was neither executed nor optimized.

## Frozen components

R2 M=2, CoPS, VAE, DFG, forward Factor Bank geometry, true residual, ACT, rho=.05, data, optimizer/LR.

## Falsification result

`FACTOR_ROLE_DECOMPOSITION_REDESIGN_REQUIRED`: no batch6 benchmark, E20, medical inference, or final scripts are authorized. A future attempt must establish a semantically stable factor-role decomposition before revisiting routing.
