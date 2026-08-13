# Phase4 R2 Role-Decomposition Failure Checkpoint

## Terminal State

`FACTOR_ROLE_DECOMPOSITION_REDESIGN_REQUIRED`

## What Still Works

- The R2 Factor Bank has shown useful conditional/oracle residual utility when responsibility is supplied.
- True-residual and final correction mechanics passed real strict-FP32 zero-step reconstruction checks.
- ACT is finite and rho remains fixed at 0.05.

## What Failed

- Independent Q/K Router, affine boundary residual, and direct normalized two-way Q head.
- Region-balanced Router supervision and Normal/Anomaly PCGrad.
- Offline-calibrated frozen Q head, stable region target on changing Q, and frozen/decoupled reference representation.
- DFG-posterior routing and residual self-routing.
- Intrinsic factor responsibility from the Factor Bank's own factor-state compatibility.

## Latest Intrinsic Responsibility Evidence

- TRAIN-only offline intrinsic score: std(q0-q1)=0.1403; it was numerically non-degenerate but Anomaly role1 recall was 0.
- Offline anomaly correction with ACT=1 and rho=.05: -0.008224; bootstrap 95% CI [-0.009670, -0.007893].
- Case B used balanced physical Normal/Anomaly CE over intrinsic factor-state compatibility, `p_route=stopgrad(softmax(q))`, and differentiable residual `delta` values. The legacy Q/K Router was frozen and not executed.
- Strict-FP32 zero-step passed: finite forward/backward, exact residual/correction reconstruction, rho=.05, zero optimizer state, intrinsic factor gradient norm 0.005219.
- Fresh short64 reached both regions across 11 optimizer updates, but anomaly p(role1) fell 0.5113 -> 0.3185 and anomaly role1 agreement fell 0.8333 -> 0.0000; Normal ACT=1 gain became negative.

## Scientific Interpretation

The current two-role Factor Bank can produce useful role-specific residuals when responsibility is known, but neither external nor intrinsic responsibility has remained identifiable and useful under training. The next research question is the definition/decomposition of the roles/factors themselves, not another Router head.

## Reproduction Artifacts

- `runs/p1_v84a_gpu/intrinsic_factor_responsibility_seed0/offline/intrinsic_factor_responsibility_audit.json`
- `runs/p1_v84a_gpu/intrinsic_factor_responsibility_zero_step_seed0_attempt8/backward_probe.json`
- `runs/p1_v84a_gpu/intrinsic_factor_responsibility_short64_seed0_attempt2/region_router_short_gate.json`
- `FACTOR_ROLE_DECOMPOSITION_SCIENTIFIC_DIFF.md`

## Next Authorized Direction

Factor/role decomposition redesign.
