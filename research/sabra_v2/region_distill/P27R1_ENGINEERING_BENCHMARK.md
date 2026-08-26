# P27R1 Engineering and Cache Runtime Evidence

Status: `PASS`

The scientific run reused the preregistered exact cache path and the
pre-existing measured engineering comparison. No scientific parameter or
protocol change was made during supervision.

## Measured cache performance

| Measurement | Result |
|---|---:|
| Uncached median sec/step | 0.2977213231 |
| Cached median sec/step | 0.0329064033 |
| Measured speedup | 9.0475194260x |
| Frozen forward median sec/sample | 0.2554102070 |
| Cache read median sec/sample | 0.0019367030 |
| Teacher build median sec/sample | 0.0063014342 |
| Cached GPU peak allocated | 137,139,200 bytes |
| Uncached GPU peak allocated | 2,182,936,576 bytes |

The exactness evidence reports bit-exact features, native logits, teacher
targets, loss, gradients, and optimizer-updated adapter state for the bounded
comparison. The frozen scientific loader remained `num_workers=0`, pinning
off, blocking transfer, seed `0`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

## This run's recorded timings

- Tier-A cache build: 720.608997 s.
- All 12 Tier-B cache builds: 353.677442 s combined.
- Adapter training: 23,414.075892 s combined across 12 folds.
- Held prediction: 71.651518 s combined across 12 folds.
- Reported scientific wall time: 25,124.037862 s (6h 58m 44.038s).
- The remaining recorded wall time includes scoring, aggregation, and process
  overhead; no training or fitting occurred during scoring.

Cache storage provenance was 27,348,348,720 bytes projected for Tier A,
25,532,830,840 bytes projected for Tier B, and 52,881,179,560 bytes total.

Source/cache safeguards recorded zero MVTec reads and zero Medical reads.
Tier-B manifests recorded source-only inventories and zero held-mask reads.
