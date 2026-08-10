# P1-v8.4-A Frozen Router Replay Root Cause

The failed replay must not be retried automatically. This investigation used only existing source, committed audit artifacts, the failed-replay evidence, logs, and provenance.

## Data scope

Both audits use VisA/train, seed 0, batch size 1, the same seeded shuffled DataLoader, 300 batches, the historical canonical root `/workspace/data/med_visa/data`, and unchanged image/mask preprocessing plus `local_mask_valid` construction. Reconstructing the DataLoader order without loading model data reproduces all 300 historical indices exactly.

## Executable margin formula

The authoritative audit source at `948bc3d` first filters all tensors by `valid`, then computes:

```text
best = max(gain_rel)
second = second-largest(gain_rel)
margin_rel = (best - second) / max(abs(best), 1e-12)
eligible = best > 0 and margin_rel > 0.10
```

The new tool computes the same quantities in their original tensor shape and adds `valid` to the final predicate. This is exactly equivalent to the historical valid-patch prefilter; a focused regression test compares eligible and winner counts directly.

Historical expected final eligibility is overall=1,071,260, normal=1,066,750, anomaly=4,510. Historical anomaly winners are F2=2,215, F3=1,895, F4=400.

## Provenance limit and decision

The actual replay reached 300 batches but its combined-eligibility assertion failed. The original new audit tool raised before serializing its observed condition counts or winner counts. The historical audit does not retain raw gains, individual condition masks, or a content hash of the dataset files. Therefore existing evidence cannot determine whether the unobserved divergence originated in raw gains or unrecorded input-byte drift.

This rules out a formula/threshold-only mismatch and does not prove a data-semantics mismatch. The required classification is:

`HISTORICAL_ROUTER_MARGIN_PROVENANCE_UNRESOLVED`

## Audit-only correction prepared

The audit now records compact condition counts and winner counts by region before reporting an invariant failure. It also uses anomaly-region F2/F3/F4 counts for the tau-usability contract. These changes do not alter model, data, Router, ACT, factor, or training semantics. No replay was launched after this correction.

Another replay requires explicit user authorization after review.

EXIT_FOR_DISCUSSION
