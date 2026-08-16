# P5B candidate preregistration

## Frozen candidate

`P5B_POSITIVE_ONLY_MIN_PROJECTION` is frozen after the pushed R0 evidence boundary
`8865489a7aad490d218886e8ec534187f9f70e12`. The candidate name is the
predetermined conditional branch, not a choice made by comparing candidate
performance on R0 outcomes.

The future comparison has exactly these conditions:

- `C0`: unchanged Phase2B.
- `P5`: the frozen R0 aligned GT-free selector followed by the positive-only
  minimum projection below.
- `P5_SHIFT`: the identical selector/action pipeline using only the frozen
  shifted E, E_stage, and E_LOO control maps.

There is no sweep, learned calibration, threshold search, or post-result
formula change.

## Selector and action contract

The selector uses the exact R0 protocol: top `ceil(0.20*1369)` D_rank risk,
exact B2 within-image 10-bin m_bar and D_rank cells, valid-reference gating,
same-image K=8 frozen peers, strict E relation, 3-stage and 8-view unanimity,
minimum-cost deterministic sorting, disjoint one-pass selection, and no
cascade. Shift changes only the E maps; m_bar, D_rank, validity, cells, and
peer process remain fixed.

For every selected strict base inversion `m_bar_i < m_bar_j`, define
`g = m_bar_j - m_bar_i > 0` and apply the non-strict constraint
`m_prime_i >= m_prime_j` with:

```text
delta_i = g
delta_j = 0
m_prime_i = m_bar_j
m_prime_j = m_bar_j
```

All unrelated coordinates are unchanged. The same nonnegative `delta_i` is
added to the anomaly native logit channel at all three stages. The normal
native logit channel is unchanged. `D_rank` and E are not recomputed, no
epsilon or margin hyperparameter is used, and no second pass is allowed.

## Deployment and readout

The exact deployment is native 37x37 -> Gaussian blur 7x7 sigma1 -> bilinear
resize with `align_corners=True` -> stage mean -> softmax. The candidate
readout must report pixel AP/AUROC, native patch AP, Normal FPR at C0 tau95 and
tau99, normal mean/p99/max anomaly probability, negative pairwise ranking risk,
selected rescue/break, action count, native action magnitude, spatial support,
native-versus-deployed behavior, aligned-versus-shifted behavior, and
per-class results with class-bootstrap CIs.

No claim of AP improvement, ranking guarantee, causal isolation, or medical
transfer is preregistered.

## Readiness boundary

Before one full evaluation, the isolated implementation must pass deterministic
unit tests, static predictor-integrity checks, and a zero-GT replay over the
already finalized R0 cache. A failure means `P5B_IMPLEMENTATION_NOT_READY` and
no full candidate evaluation.
