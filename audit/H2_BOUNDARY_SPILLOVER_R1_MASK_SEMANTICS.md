# H2 Boundary Spillover R1 — Mask Semantics

The audit uses the exact existing dataset mask path and the existing
morphology semantics; no morphology-radius sweep or new mask rule is
introduced. Source masks are resized to 518x518 with nearest-neighbor
interpolation by `BaseSingleClassDataset`, then binarized. Native-stage ground
truth is the exact existing nearest-neighbor resize of that source mask to the
37x37 patch grid (`F.interpolate(..., mode="nearest")`).

For every resolution, the existing 7x7 all-ones structuring element is used
exactly once: `boundary = mask & ~binary_erosion(mask)`, `interior =
binary_erosion(mask)`, `near_background = binary_dilation(mask) & ~mask`, and
`far_background = ~binary_dilation(mask)`. Normal images have an empty anomaly
mask and therefore contribute only to positive/negative or global metric
calculations, not anomaly-region interior/boundary/distance summaries. Empty
regions are reported as null.

The 1-pixel distance bin is the standard Euclidean distance-transform interval
`(0, 1]`; subsequent fixed bins are `(1,2]`, `(2,4]`, `(4,8]`, `(8,16]`, and
`(16, infinity)`.
