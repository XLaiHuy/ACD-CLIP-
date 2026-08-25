# P25R3 Numerical Contract

## Exact objective and gradient

The authoritative beta-space objective is the P25R2 expression:

`L(beta) = mean(w * logaddexp(0, -D beta)) + 0.5 * beta^T beta`,

where `w` is the original pair-benefit weight divided by its original mean.
Its gradient is:

`g_beta = mean(-w * sigmoid(-D beta) * D, axis=0) + beta`.

For active coordinates, `beta=Tz`, `T=diag(1/s)`, and
`g_z=T^T g_beta`. Objective and chain-rule gradient parity are required on
multiple deterministic nonzero vectors with error at most
`1e-12 * max(1, reference magnitude)`.

## Preconditioner and inactive dimensions

`s_k=max(max(abs(D[:,k])),1)` uses source-training design only. A column is
inactive only if its exact maximum absolute value is zero. Its beta is fixed to
zero, which is the exact optimum for a loss-independent coordinate under the
original positive L2 penalty. No outcome-dependent preconditioner selection is
permitted.

## Regression and optimality

The pre-marker known-failure fixture reconstructs the chewinggum-excluded
source design without reading chewinggum targets. Historical beta=0 must fail
the certificate with relative gradient 1. The recovered solver must either
return a nonzero certified beta or explicitly reject the fit.

The certificate is `solver_success && finite && final<=initial &&
relative_gradient_inf<=1e-7`. A constant prediction from a certified fit keeps
Pearson/Spearman null and follows the inherited aggregation contract.
