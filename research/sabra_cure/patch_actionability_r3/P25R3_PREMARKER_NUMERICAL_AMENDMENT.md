# P25R3 Pre-Marker Numerical Amendment

## Status and reason

This amendment was frozen before `ATTEMPT_STARTED.json`, before any held Q1
prediction, and before any P25R3 scientific metric. The published preregistered
SciPy L-BFGS-B implementation reproduced the correct P25R2 source-only design
and rejected its own result under the already-frozen original-beta certificate.
It reduced the relative beta-gradient infinity norm only to
`1.2136919862692325e-3` in 1,000 iterations. Deterministic same-settings
continuations stalled at about `3.593e-6`, above the frozen `1e-7` requirement.

This is numerical engineering evidence only. No held target, held prediction,
held metric, gate, or Q2 outcome was read.

## Frozen replacement solver

The one production optimizer is deterministic damped Newton in the already
frozen active z coordinates. It uses float64, zero initialization, and the
exact transformed gradient and Hessian of the unchanged beta-space objective.
For `A = D[:,active] / s[active]`, `p=sigmoid(-A z)`, and normalized frozen
pair weights `w`, the Hessian is:

`H_z = A.T @ ((w*p*(1-p))[:,None] * A) / n + diag(1/s_active^2)`.

Each Newton direction solves `H_z d = -g_z` with `numpy.linalg.solve`. A
deterministic Armijo line search starts at step 1, uses constant `1e-4`, and
halves until acceptance; a step below `2^-30`, a failed solve, any nonfinite
quantity, or failure to certify within 50 Newton iterations is an engineering
stop. There is no fallback solver, held-dependent choice, or solver zoo.

The original-beta certificate remains exactly the published threshold:

`||g_beta(beta)||_inf / max(1, ||g_beta(0)||_inf) <= 1e-7`.

On the source-only chewinggum-excluded regression, this solver reached
`1.913733309710175e-11` after 11 evaluated states. Its recovered beta norm was
`0.2611863873446141`. The historical beta=0 state remains rejected with
relative gradient 1.

## No science change

The target, pair inventory and weights, feature scaling, beta-space objective,
L2, prediction definition, metrics, gates, routing, folds, leakage rules, and
undefined-correlation semantics are unchanged. This amendment changes only the
algorithm used to locate and certify the same convex optimum.
