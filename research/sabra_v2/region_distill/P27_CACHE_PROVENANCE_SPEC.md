# P27 Cache Provenance Specification

## Tier A: reusable GT-free frozen cache

Tier A is class-sharded under `/workspace/p27_cache_v1/tier_a/<class>/` and
uses contiguous NumPy `.npy` memmaps. Each shard contains only `seg_features`
with sample shape `[3,1369,768]` and `native_logits` with sample shape
`[3,1369,2]`, both in their original `float32` dtype. It contains no labels,
masks, teacher targets, fitted statistics, thresholds, or normalization.

Tier A is safe to reuse across all LOCO folds because both tensors are
deterministic functions only of the image and the immutable P26 checkpoint,
CLIP asset, and Phase2B configuration. Stable sample IDs are
`<class_name>:<image_path>`, and each class manifest fixes their order.

## Tier B: fold-local source supervision

Tier B is sharded by held fold under
`/workspace/p27_cache_v1/tier_b/<held_class>/`. It contains processed source
masks `[1,518,518]` and source teacher region targets `[9,9]`, both `float32`.
Its manifest stores exactly the 11 source classes and exact source sample IDs.
Validation rejects any requested row whose class is held or whose ID is absent
from that fold's source-only manifest. The held-mask-read counter must be zero.

## Completion and rejection

Each shard is written to a UUID-named `.incomplete` sibling. Array contents are
flushed, their byte sizes and SHA-256 hashes are recorded, and `manifest.json`
is written last. The directory is then atomically renamed. Loaders reject
missing manifests, non-`COMPLETE` status, missing/truncated arrays, unexpected
shape or dtype, duplicate/missing/reordered sample IDs, wrong source inventory,
and any provenance mismatch.

Every manifest fixes:

- schema, tier, implementation version, completion status;
- class or held fold, sample IDs/count, tensor names/shapes/dtypes/file hashes;
- P26, CLIP, Phase2B config, and metadata SHA-256 digests;
- parent execution SHA and scientific execution-base SHA;
- explicit GT/supervision firewall fields.

No cache blob is stored in Git.
