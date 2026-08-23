# P17 Process Isolation Contract

For every held class, the parent spawns exactly one worker, waits for exit zero,
validates that worker's compact fold checkpoint hashes, confirms the worker PID
is gone, records parent RSS, then moves to the next frozen class. At most one
worker exists at a time.

Workers reconstruct one fold, atomically persist only their own compact fold
artifacts/status/log, and never aggregate 12-fold metrics or inspect previous
fold outcomes. The parent holds no score map, mask tensor, shard, grouped AP
structure, or outer-fold result.

Frozen engineering gates: child peak RSS <= 14 GiB; parent RSS after each child
<= warmed parent RSS + 512 MiB; no material monotonic parent growth; no live
worker PID after exit. Process exit—not allocator trimming—is the authoritative
memory-lifetime boundary.
