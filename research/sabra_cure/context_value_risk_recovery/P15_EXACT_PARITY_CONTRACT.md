# P15 Exact Parity Contract

Frozen P14 implementation hash:
`d83c59c7e52b21022c90708198048f049bd4e4e46ff443a67adf0c524414273f`.

Allowed transformations are AP-only computation, immutable cache reuse,
exact float32-score grouped-count deltas, cached composition of SAFE20/E40
maps, and bounded deterministic threading. They must reproduce the frozen
reference calculation; no score quantization, approximation, target change,
or result-dependent method choice is allowed.

Reference and optimized engines are compared on synthetic random, all-tie,
repeated-float32, normal/anomalous switch, no-change, large-crossing, and
created/deleted score-group fixtures. Real fixtures are frozen before values:
classes `candle`, `capsules`, `cashew`, first and last image in each class.
Only AP values, absolute error, runtime, and speedup are retained for the real
benchmark—not target sign/value interpretation.

Acceptance is `abs(reference_pAP - optimized_pAP) <= 1e-12` for every real
fixture, with exact float32 tie-group semantics. Multi-image SAFE20/E40 policy
composition, cached/uncached deploy, and one-worker/N-worker equality are also
mandatory.
