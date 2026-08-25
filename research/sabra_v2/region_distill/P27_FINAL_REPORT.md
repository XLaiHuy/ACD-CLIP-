# P27 FINAL REPORT

## Engineering

- Parent SHA: `1151373f2c4968268f52cdc3e538c7ebcef7b8f0`
- Recovery branch: `research/p27-cache-performance-recovery-v2`
- Scientific execution-base SHA: `b1277c6ea30a749f8218980f140227106e7ed77a`
- Cache architecture: cross-fold class-sharded GT-free Tier A plus fold-local
  source-only Tier B
- Cache format/dtype: contiguous NumPy `.npy` memmap, original `float32`
- Projected full cache size: `52,881,179,560` bytes (`49.2494 GiB`)
- Actual scientific cache size: `0` tensor bytes; failure occurred before the
  first cache sample
- Uncached median sec/step: `0.2977213231`
- Cached median sec/step: `0.0329064033`
- Measured speedup: `9.0475x`
- Projected Tier-A build time: `552.20 s`
- Projected Tier-A plus training runtime: `4.50 h`
- Actual attempt runtime before stop: approximately `3.59 s`
- Uncached/cached peak GPU allocated: `2,182,936,576` / `137,139,200` bytes
- Engineering peak host RSS: `3,171,048 KiB`

## Scientific

- Attempt UUID: `60dd4d8d-15cd-403e-b2b3-4b38f4e7da1a`
- Attempt count: `1`
- Completed folds: `0`
- Immutable held predictions: `0`
- Scored folds: `0`
- Native/P27 macro and per-class metrics: unavailable; scoring was never
  reached

## Audit

- Held GT reads before scoring: `0`
- Held mask reads before scoring: `0`
- MVTec reads: `0`
- Medical reads: `0`
- Phase2B optimization steps: `0`
- CLIP optimization steps: `0`
- RegionResidualAdapter optimization steps: `0`
- Frozen P26, CLIP, config, and protocol: unchanged
- Attempt marker: exactly one
- Automatic rerun: forbidden by the preregistered long-run policy
- Post-audit status: `P27_ENGINEERING_STOP`

## Interpretation

### Observed

The final pre-science gates passed and the durable attempt marker was written.
The first Tier-A cache subprocess, launched in tmux `sabra`, then encountered
CUDA error 804 and reported `torch.cuda.is_available() == False`. The failure
occurred while loading the frozen checkpoint onto CUDA, before any cache tensor,
training update, held prediction, GT read, or score was produced.

### Interpretation

This is an execution-environment failure, not evidence for or against the P27
scientific hypothesis. Exact continuation is not unquestionably valid because
the process terminated immediately after the one-shot marker and before a
durable resumable cache boundary. The frozen protocol therefore forbids a
patched or restarted attempt.

### Final status

`P27_ENGINEERING_STOP`
