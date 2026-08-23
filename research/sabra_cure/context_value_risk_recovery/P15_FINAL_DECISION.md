# P15 Final Decision

## Terminal status

`P15_ENGINEERING_STOP`

Reason: `P15_ENGINEERING_STOP_UNBOUNDED_OUTER_CACHE_RETENTION`.

The one authorized P15 attempt (`54fed406-07a0-4cde-88e6-0fbe5db20a76`) was
interrupted after 1,286 seconds. It completed one outer fold checkpoint and
began the next outer fold, but retained the prior outer fold's full immutable
score-map caches in `folds`. This violates the P15 bounded per-fold cache
contract and was accompanied by resident memory growth to 14,580,356 KiB.

No partial P15 prediction, target, policy, metric, gate, or scientific outcome
is interpretable. No gate has been evaluated and no result was selected. The
attempt is consumed; no P15 resume or rerun is permitted because the engineering
implementation must not change after `ATTEMPT_STARTED.json`.

This terminal decision preserves the marker, logs, and completed atomic
checkpoint evidence. It does not alter the frozen P14 contract, the P14
engineering-stop conclusion, or any historical artifact.

Firewall/freeze: MVTec reads `0`; Medical reads `0`; additional CLIP forwards
`0`; Phase2B optimizer steps `0`.

Next allowed action: explicit user review before any separately versioned
engineering recovery is considered.
