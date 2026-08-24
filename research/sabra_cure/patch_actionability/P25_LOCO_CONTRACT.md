# P25 Nested LOCO and Pair Contract

For outer held H, all fitting, scaling, pair construction, feature/policy
selection, and threshold calibration use exactly the other 11 classes. H
labels open only after fixed held benefit scores and actions are materialized.

For every source class J in outer training, direction/harm quantities used by
the benefit model are class-excluded nested predictions: J and H are absent
from learned predictions for J. In-sample risk is forbidden.

The primary benefit model is an advantage-weighted linear pairwise ranker:
`score(x)=w^T x+b`. It uses float64, zero initialization, deterministic CPU
L-BFGS, L2=1.0, and one fixed optimizer configuration. Pairs are only within
source class, built from source-training V percentile deciles, excluding
same/adjacent deciles, deterministically balanced, and capped at 8192 per
source class. Pair weights are absolute source-class V percentile-rank gaps.
No V-magnitude regression, LambdaRank, model selection, or hyperparameter sweep
is permitted.
