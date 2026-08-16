# R0 GT-free selector reliability audit

## Decision

`terminal = R0_CONSENSUS_SELECTOR_SUPPORTED`; `selected_candidate = P5B_POSITIVE_ONLY_MIN_PROJECTION`.

R0 is a GT-free reliability audit, not a candidate evaluation. The cache was finalized and reopened before any mask/label was loaded. No AP/AUROC or candidate performance metric was computed.

## Frozen selector

Raw relations are strict base inversions within the exact B2 risk/cell population where E prefers the lower-base patch. Certification requires strict agreement at all three stages and all eight leave-one-peer-out views using the same K=8 peers. Certified relations are sorted by minimum required base-score cost and greedily made disjoint, with one accepted relation per patch and one pass. Shifted controls shift only E maps.

## Results

See `SELECTOR_SUMMARY.json` and `PER_CLASS.json` for RAW/CERTIFIED/SELECTED counts, W, rescue/damage/net, class bootstrap CIs, coverage, and shifted controls. `SELECTED_PAIRS.csv` contains post-hoc GT labels only after selection fields were frozen.

## Integrity

Unique image forwards: `2162`; physical image-forward calls: `2162`; training steps: `0`. The B3.1 pair cache was insufficient for E_stage/LOO reconstruction, so the single authorized canonical inference pass was used.

## Boundary

No candidate implementation or full VisA evaluation is performed in this R0 driver. If any R0 gate fails, the correct terminal is `R0_CONSENSUS_SELECTOR_UNSUPPORTED` and no candidate is allowed.
