# P25R3 — Exact Q1 Numerical Optimizer Recovery V1

## Authority and scope

P25R3 is a surgical numerical recovery of P25R2. It starts from forensic commit
`2d80a02366d19275b47ca5621e34090aeaf18ddd`, cites P25R2 terminal
`99ad3ab6292ca3b95fbda0cb8c6985ed9afe3253`, P25R2 preregistration
`233d16b0b29286c8ea73b7886a8d929ca263e5d7`, and P25R2 execution base
`a50d9a6d0f7c1cea007f6d18ea9cafbb6b8711d0`.

P25R2 remains `NON_INTERPRETABLE`. The forensic evidence establishes a
computation bug: chewinggum beta was exactly zero while the exact original
objective gradient at zero had L2 norm `1750236.1845595562` under a frozen
`1e-12` tolerance.

P25R3 changes only internal numerical optimization. It inherits unchanged the
P25R2 targets, 32 GT-free features and order, source-only within-class
non-adjacent-decile pair construction, pair weights, pairwise logistic loss,
beta-space L2=1, no-intercept prediction, 12-fold LOCO exclusions, Q1 metrics,
Q1 gates, undefined-correlation semantics, Q1 routing, and conditional Q2.

No label clipping, example removal, pair change, feature change, metric change,
gate change, threshold change, action change, alpha change, held-class solver
selection, or Q2 redesign is allowed. Undefined Pearson/Spearman remain null.

## Frozen numerical recovery

For source-training pair design `D`, define independently for every fold:

`s_k = max(max_i(abs(D_ik)), 1.0)`.

Columns with exact `max_i(abs(D_ik)) == 0` are inactive; their beta is fixed to
the exact original-objective optimum zero. For active columns, optimize
`beta_k = z_k / s_k`. The transformed objective is exactly the original:

`mean(w * softplus(-D beta)) + 0.5 * ||beta||_2^2`.

The penalty is evaluated through beta and is never replaced by isotropic
z-space regularization. The sole production solver is float64 SciPy
L-BFGS-B, unbounded, zero-init, analytic gradient, `maxiter=1000`,
`maxls=100`, `maxcor=20`, `ftol=1e-15`, and `gtol=1e-12`.

## Fit validity

For every fit, compute the exact original beta-space gradient `g(beta)` and
the zero-init reference gradient `g(0)`. Define:

`relative_gradient_inf = ||g(beta)||_inf / max(1, ||g(0)||_inf)`.

A fit is valid only if all quantities are finite, solver success is true,
final objective does not exceed initial objective, and
`relative_gradient_inf <= 1e-7`. This threshold is fixed before any P25R3
held prediction or metric. The known P25R2 beta=0/large-gradient state must
fail this certificate.

Persist solver status/message, iterations/evaluations, initial/final
objectives, beta norm, beta-gradient norms, relative certificate, prediction
min/max/std/unique count, preconditioner, and inactive dimensions.

## Frozen target reuse

All 12 immutable P25R2 target artifacts are reused after schema, count,
ordering, finiteness, provenance, and SHA-256 audit. P25R2 Q1 betas and Q1
scores are forbidden inputs. Q1 is recomputed for all 12 folds.

The target hashes are frozen in `P25R3_EXECUTION_CONTRACT.md`. No target
generation or deployment is authorized in P25R3.

## Routing and terminal states

One attempt begins only from a published clean execution base. After 12 valid
Q1 fits, exact inherited Q1 metrics/gates are evaluated. Q2 is entered only if
that frozen routing passes. Any invalid numerical fit is
`P25R3_ENGINEERING_STOP`; no patch or rerun follows.

No MVTec or Medical reads, additional CLIP forwards, Phase2B steps, external
validation, architecture change, or additional scientific attempt is allowed.
