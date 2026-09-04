# H2 initialization and seed-sensitivity forensics

## Validity boundary

Seed 0 is a discovery/factorial run. Seeds 1 and 2 are fresh H/A
confirmatory runs from their own shared E1 checkpoints, but both fail the hard
validity rule because skipped nonfinite-gradient steps make H and A have
different successful optimizer-step counts. No Seed-1 or Seed-2 Medical or
MVTec target metric exists.

## Observed source-side differences

| arm | Seed 1 loss E2 -> E15 | Seed 2 loss E2 -> E15 | gradient-skip pattern |
|---|---|---|---|
| H | 1.144242 -> .769624 | 1.185225 -> .762068 | 3 skips in each seed, different epochs |
| A | 1.193046 -> .946489 | 1.196296 -> .941572 | 2 skips in each seed, different epochs |

The repeated skip counts across seeds are evidence of a reproducible validity
hazard. The differing epochs and the small source-loss differences show that
the exact trajectory is seed-sensitive, but they do not provide a valid
between-seed target variance estimate.

## What could not be measured

The archive does not contain a parameter-distance table between shared E1 and
the final checkpoints, representation similarity, target prediction-map
similarity, or valid H/A target metrics for Seeds 1 and 2. Checkpoint files are
preserved, but loading them to invent a post-hoc metric is outside this
forensic phase and would not repair the invalid optimizer histories.

Therefore:

- `INITIALIZATION_SENSITIVITY=NOT_ESTIMABLE` from the current valid evidence.
- `ANCHOR_VARIANCE_EFFECT=NOT_ESTIMABLE`.
- `ANCHOR_REPLICATION_SUPPORT=NOT_CONFIRMED`.

The correct scientific statement is that multi-seed robustness is unresolved,
not that A is refuted. A source-only numerical-stability trace and fresh valid
replication are required before estimating variance or selecting a robust
winner.
