# P16 Exact Parity Contract

P16 reuses P15's exact float32-score grouped AP engine and preserves P14 source
`tools/sabra_cure/context_value_risk.py` SHA-256
`d83c59c7e52b21022c90708198048f049bd4e4e46ff443a67adf0c524414273f`.

Reference and recovery AP must agree exactly where achievable; the frozen real
fixture acceptance criterion remains maximum absolute AP error `0.0` (and no
more than `1e-12` only if a documented platform-level tie issue makes equality
impossible). SAFE20, EXPAND40, target, features, ridge, q set, alpha, source
selection, comparators, gates, and float/tie policy are unchanged.

Pre-marker fixtures cover random, all-tie, repeated-float32, normal/anomaly,
no-change, large-crossing, created/deleted group, multi-image policy, compact
checkpoint reload, one-worker/N-worker determinism, and two sequential
synthetic fold lifecycles. No real P16 outer scientific fold or P14 gate is
allowed in an engineering fixture.
