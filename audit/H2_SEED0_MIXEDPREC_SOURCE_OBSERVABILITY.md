# H2 Seed0 mixed-precision source observability

`HISTORICAL_SEED0_SOURCE_AUDIT=UNAVAILABLE`.

The recorded historical A E15 checkpoint is
`frozen_seed0_final/A/adapter_15.pth` (SHA256
`727dc4813db1ef0c5a6db3a4cf15e916ad1413dcf629913358419d2734edecfe`), but
the checkpoint is absent from the repository, `/workspace`, and mounted
historical locations. Repository provenance identifies the Seed0 contract,
but cannot supply model weights for a source-only replay. No source metrics
were inferred or guessed from historical target exports.

## Recorded provenance

- Seed: 0; epoch: 15; VisA manifest SHA:
  `468463d2d6234fa7537c6da32b027758527676a12a54a4028c5a282cdd726842`.
- CLIP SHA: `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`.
- Safe Anchor lambda: `0.0021633926715180626`; family cap rho: `0.10`.
- Historical precision policy: FP16 autocast with GradScaler and legacy local
  FP32 repair/islands. Historical E2–E15 had two non-finite-gradient skips,
  so it does not meet the current BF16 validity bar.

This unavailable source replay does not block the authorized single Seed0 BF16
disambiguation run; it prevents only a source-mechanism comparison with the
historical mixed-precision weights.
