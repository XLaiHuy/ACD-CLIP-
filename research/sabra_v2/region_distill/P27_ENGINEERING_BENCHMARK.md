# P27 Cache Performance Recovery — Engineering Evidence

Status: `PASS`

The bounded comparison used five real P26/VisA source samples after one warmup
on the RTX 3070 Ti. It compared the original RGB-to-Phase2B training path with
the exact memmap-backed P27 path. Medians, not single timing samples, are
reported.

## Exactness

- `seg_features`, `native_logits`, processed masks, and teacher region targets:
  bit-exact; maximum absolute and relative difference `0`.
- Total loss: bit-exact; difference `0`.
- Adapter gradients: bit-exact; difference `0`.
- Optimizer-updated adapter state: bit-exact; difference `0`.
- Deterministic sample order and checkpoint state were also bit-exact across
  loader-worker configurations.

The canonical CUDA deployment backward reports that reflection padding and
adaptive average pooling do not have declared deterministic kernels. The
runner therefore uses PyTorch's strongest available warn-only deterministic
policy plus `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Despite those declarations, the
measured cached/uncached gradient and optimizer parity was bit-exact.

## Timing and memory

| Measurement | Result |
|---|---:|
| Uncached median sec/step | 0.2977213231 |
| Cached median sec/step | 0.0329064033 |
| Measured speedup | 9.0475x |
| Frozen forward median sec/sample | 0.2554102070 |
| Cache read median sec/sample | 0.0019367030 |
| Teacher build median sec/sample | 0.0063014342 |
| Adapter forward/backward/update median sec/step | 0.0175816142 |
| Uncached peak GPU allocated | 2,182,936,576 bytes |
| Cached peak GPU allocated | 137,139,200 bytes |
| Peak host RSS | 3,171,048 KiB |
| Direct sequential disk write/read | 5.3 / 7.8 GB/s |

The exact Tier-A payload is 12,649,560 bytes/sample. For 2,162 records, Tier A
projects to 27,348,348,720 bytes. All 12 fold-local Tier-B source caches
project to 25,532,830,840 bytes. Total projected cache storage is
52,881,179,560 bytes (49.2494 GiB), leaving far more than the required 80 GiB
reserve on the measured 457 GB-free filesystem.

Projected Tier-A build time is 552.20 seconds. Tier-A build plus all 475,640
training steps projects to 4.50 hours before the comparatively small Tier-B,
prediction, and scoring overheads, comfortably below the 18-hour stop gate.

## Frozen loader configuration

One bounded refinement compared ten exact cached updates:

| Workers | Pin/non-blocking | Median sec/step |
|---:|---|---:|
| 0 | off/off | 0.0202333881 |
| 2 | on/on, prefetch 2 | 0.0445898606 |
| 4 | on/on, prefetch 2 | 0.0434599388 |

All three produced bit-identical adapter state. The scientific configuration
is frozen at workers `0`, pinning off, and blocking transfer.
