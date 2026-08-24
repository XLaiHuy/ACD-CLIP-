# P21 Protocol Interpretation Note

This note closes operational degrees of freedom that were not numerically
spelled out in the published P21 preregistration. It is published before any
P21 outcome calculation, attempt marker, or execution-base freeze. It changes
no P21 action, target, feature family, metric, gate, fold, or route.

## Linear RankNet probes

P1 and P2 minimize the sum over every eligible within-source-class pair of
`logaddexp(0, -label * ((x_i - x_j) @ w)) + 0.5 * ||w||^2` after the frozen
source-train-only median/IQR transform. The intercept is fixed at zero because
it cancels identically in every pairwise difference and is unregularized.

The deterministic optimizer is SciPy `L-BFGS-B`, initialized at an all-zero
float64 vector, with `maxiter=500`, `maxls=50`, `gtol=1e-10`, and
`ftol=1e-15`. There is no learning-rate, batch, pair-sampling, class
reweighting, early-selection, or hyperparameter search. A non-success return,
non-finite parameter, or non-finite prediction is an engineering stop.

## F1 deterministic rank details

For a map of `N` pixels, stable ranks use descending `np.argsort` with
`kind="mergesort"` and rank positions `0..N-1`. The top decile is exactly the
first `ceil(0.10*N)` stable positions. The fourth F1 feature is the frozen
stable-rank Spearman correlation of the flattened native and action maps; a
constant input is an engineering stop rather than a silent zero correlation.

All four F1 statistics are computed from the frozen float32 score maps, then
stored/trained as float64. This note preserves inference-time GT-freedom: no
mask, target, utility, or actionability label enters F1.
