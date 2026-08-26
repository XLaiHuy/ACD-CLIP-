# P30 Final Report — Directional Distillation

## Decision

**REJECT — scientific stop at Stage 2.** The preregistered one-class candle
gate failed, so the fixed four-class subset and full 12-class attempt were not
run. No full execution marker was created and no scientific rerun is allowed.

- P30 UUID: `71a16efe-2388-458a-9106-bd87f882805a`
- Preregistration SHA-256:
  `86ede5f23c7edd50ed848fa1e6c6aab987e7a9ded35a1c7c3a57f12ec1badde6`
- Qualification execution base: `2360a04cfe2624f6794dc666299a78dab4e17ae0`
- Full marker: **absent**
- Full 12-class result: **not claimed**

## What was tested

The single preregistered objective compared the staged student correction with
the cached teacher correction using only per-sample directional cosine over
243 normalized coordinates. The P29 adapter, Tier-A/Tier-B cache, LOCO split,
FP32 policy, AdamW schedule, and unchanged P26 deployment were retained.

Static qualification, 57 selected relevant tests (including 15 new P30 tests),
synthetic directional/gradient checks, one-step smoke, checkpoint reload, and
the 40-step engineering profile passed. The one-class candle run completed
39,240 optimizer steps with finite gradients, zero teacher update, and zero
held GT/mask reads before prediction freeze.

## Stage 2 result

| Metric | P30 | Frozen P29 | Result |
|---|---:|---:|---:|
| Pixel AP | 0.144618064 | 0.490503231 | −0.345885167 vs P29 |
| Pixel AUROC | 0.972904419 | 0.970040297 | +0.002864122 vs P29 |
| Directional cosine | 0.736923574 | 0.708549174 | +0.028374400 |
| Sign agreement | 0.569567901 | 0.565493827 | +0.004074074 |
| Pearson correction correlation | 0.378208786 | 0.769957204 | −0.391748418 |
| Spearman correction agreement | 0.714233750 | 0.717418156 | −0.003184406 |
| Mean absolute residual | 1.542490326 | 2.036682759 | lower, but unstable tail |
| Residual q99 absolute | 25.929798947 | 4.321676936 | materially inflated |
| Normal score q99 shift | 0.998690784 | 0.000001159 | materially inflated |

P30 therefore improved the intended directional cosine and slightly improved
sign agreement, but the directional-only objective left correction magnitude
unconstrained enough to create a severe tail and normal-score inflation. The
result is a direct negative answer to the candidate-1 mechanism test: removing
the magnitude term did not preserve useful deployment behavior on candle.

## Gradients and speed

- All recorded gradients were finite; the one-class gradient diagnostic had
  `nonfinite_count_max = 0` and sampled step 1 plus every 1,000-step interval.
- The zero-output smoke had finite, nonzero student-output gradient; the
  one-class student parameter delta was L2 `48.75715841001832`; teacher delta
  was exactly `0.0`.
- The 40-step cached engineering profile measured median P30 step time
  `0.006899061845615506` seconds versus frozen P29
  `0.010768339969217777`, or `−35.9319833387777%`. This speed benefit does not
  offset the failed stability/performance gate.
- No new CLIP or Phase2B forwards occurred. Inference was not evaluated as a
  scientific full-run claim because Stage 2 stopped the protocol.

## Audit and preservation

All candle held predictions were frozen before scoring; scoring read masks only
after the gate. No MVTec, Medical, P29, or P27 scientific rerun occurred. The
failed qualification evidence, checksums, training metadata, metrics,
transfer diagnostic, stability diagnostic, and scoring gate are retained under
`P30/qualification/stage2_one_class/`. Large immutable prediction tensors are
kept locally and their hashes are recorded in
`P30_LARGE_ARTIFACTS_SHA256.txt`; they are intentionally not added to Git.

Authoritative machine-readable audit:
`P30_POST_RUN_AUDIT.json`.
