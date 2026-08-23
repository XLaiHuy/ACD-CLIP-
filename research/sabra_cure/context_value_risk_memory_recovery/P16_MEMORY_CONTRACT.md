# P16 Memory Contract

Only one outer-fold full working set may be reachable at a time. After an outer
fold has atomically persisted its required fold artifacts, validated their
hashes, and contributed compact aggregation values, no full score map, mask,
grouped AP structure, deployment tensor, temporary rank buffer, or nested base
object belonging exclusively to that completed fold may remain reachable from
live experiment state.

Permitted cross-fold state is limited to compact scalar metrics, held IDs,
selected q/threshold, compact model parameters, action/expand masks required by
audit, output hashes, attempt identity, and deterministic provenance. Required
checkpoints contain compact target/features and fold artifacts, never a score
map cache.

`finalize_outer_fold()` must atomically persist and hash artifacts, update
progress and `memory_progress.jsonl`, explicitly release fold-local owners,
run `gc.collect()`, and release idle CUDA allocator memory where applicable.
Before allocation for fold k+1, a direct cache-owner assertion requires zero
completed-fold cache leases.

Frozen engineering gates:

- M1: `RSS_post_finalize <= RSS_pre_fold + 1 GiB` for every completed fold.
- M2: no completed-fold cache lease is live before the next fold begins.
- M3: peak RSS is at most **14 GiB**. The recommended 10 GiB limit is
  objectively unsuitable for this engine because P15's single active outer
  fold reached about 11 GiB before cross-fold retention; 14 GiB preserves a
  measurable safety margin while remaining below the observed 13.9 GiB fault.

All memory instrumentation is observational only and cannot alter science.
