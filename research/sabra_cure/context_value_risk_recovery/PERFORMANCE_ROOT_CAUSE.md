# P14 Performance Root Cause

P14 stopped for engineering efficiency, not scientific performance. Static
inspection of the frozen reference implementation
`tools/sabra_cure/context_value_risk.py` (SHA-256
`d83c59c7e52b21022c90708198048f049bd4e4e46ff443a67adf0c524414273f`)
establishes the following causes without calculating a P15 outcome.

- **B1/B2:** `image_targets()` calls `exact_metrics()` once per image after
  replacing that image's scores in a full class score tensor. This performs a
  full-class score ordering for every image target.
- **B3/B4:** the target needs only pAP, but calls `exact_metrics()`, which
  additionally computes pAUROC and delegates score grouping/sorting to the
  generic metric path.
- **B5/B6:** `deploy()` reopens each class NPZ and reloads masks every time it
  is invoked. SAFE20 and EXPAND40 maps are recreated instead of being retained
  as immutable per-class objects.
- **B7:** `source_selection()` invokes `deploy()` for SAFE20 and every
  candidate image-policy composition although these policies are selections of
  already determined SAFE20/EXPAND40 image maps.
- **B8:** the inner image-target loop is serial; the observed P14 run used one
  saturated CPU core in this dominant path.

The recovery will use an AP-only exact grouped-count delta evaluator, bounded
per-class cache, cached score-map composition during source selection, and
fixed bounded threading. No scientific contract field may change.
