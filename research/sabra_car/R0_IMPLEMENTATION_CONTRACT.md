# R0 Signed-Direction Implementation Contract

Status: FROZEN BEFORE IMPLEMENTATION AND RESULTS

Timestamp: 2026-08-23T01:15:03+07:00

This document resolves operational details left implicit by the master
preregistration. It does not change its alpha grid, models, metrics, gates, or
fallback logic.

## Inputs and provenance

- Use the immutable canonical VisA cache under
  `/home/ai4/caohuy/acdclip_runs/canonical_sabra_v1_seed0/sabra_source`.
- Validate all twelve shard hashes, the 2,162-record count, class inventory,
  checkpoint, CLIP, config, metadata, and six implementation hashes against the
  manifest before opening source GT.
- Validate BASE_SHA ancestry and the exact E10 checkpoint SHA256.
- Open only VisA masks through `dataset/hub/VisA.jsonl`; Medical reads are zero.
- Native cached pixel probabilities must match zero-delta canonical deployment
  on a deterministic sample before scientific measurement.

## Direction and conditions

For each image and patch, compute `u=-dL/d(delta)` at zero using the exact
canonical Focal+Dice loss and shared abnormal-channel intervention. Define
BOOST, KEEP, and SUPPRESS with epsilon `1e-8` exactly as preregistered.

For each alpha in `{0, 0.125, 0.25, 0.5, 1.0}`:

- native: zero delta;
- positive-only: `+alpha*s_m` on BOOST and zero otherwise;
- signed: `alpha*s_m*sign(u)` on BOOST/SUPPRESS and zero on KEEP.

`s_m=19.840438842773438`. Normal-channel delta is exactly zero and one patch
delta is broadcast identically over all three stages.

The selected alpha maximizes signed macro class pAP. Exact ties select the
smaller alpha. The matched positive-only comparator always uses this same
selected alpha. No positive-only-specific alpha is selected.

## Signed-radius coordinate oracle

For each non-KEEP patch `p`, evaluate the five coordinate counterfactuals from
the same native baseline while every other patch remains zero:

`delta_p in sign(u_p)*s_m*{0,0.125,0.25,0.5,1.0}`.

Choose the candidate with the lowest exact full-image canonical Focal+Dice
loss; exact ties choose the smaller magnitude. After all coordinate choices are
made independently from the native baseline, deploy their combined radius map
once and report its exact metrics. This is explicitly an oracle headroom
diagnostic, not a claim that the jointly combined map minimizes loss.

The implementation may exploit linear deployment, compact influence support,
batching, or algebraic loss updates, but must pass parity against direct
canonical deployment/loss for deterministic synthetic and real samples.

## Metrics and diagnostic definitions

- pAP and pAUROC are exact global-pixel metrics computed separately per class;
  macro values are the unweighted mean of the 12 class values.
- Canonical loss is accumulated as the sample-weighted mean over images.
- BOOST/KEEP/SUPPRESS rates use all `2162*1369` source patches.
- sign reversal rate is `SUPPRESS/(BOOST+SUPPRESS)`; if the denominator is zero,
  it is null and the direction gate fails.
- Class breadth is the number of classes whose signed pAP is strictly greater
  than matched positive-only pAP.
- pAUROC safety is signed macro pAUROC minus matched positive-only macro
  pAUROC and passes at `>=-0.50 pp`.
- For each image containing both normal and anomalous mask pixels, compare
  signed-selected-alpha against native. AP/loss differences with absolute value
  `<=1e-12` are ties and excluded from the four requested directional counts;
  ties are reported separately. The four counts are loss-down/AP-up,
  loss-down/AP-down, loss-up/AP-up, and loss-up/AP-down.

## Decision logic

- G0: every provenance, parity, finite-value, no-training, and no-Medical check
  passes.
- G1: signed minus matched positive-only macro pAP is at least `+1.00 pp`.
- G2: signed pAP is strictly higher in at least `8/12` classes.
- G3: signed minus matched positive-only macro pAUROC is at least `-0.50 pp`.
- R0 PASS requires G0-G3.
- R0B is triggered only when signed minus native macro pAP is at least `+1.00
  pp`, at least `8/12` classes improve over native, signed minus native macro
  pAUROC is at least `-0.50 pp`, R0 does not pass, and at least half of eligible
  non-tied images have discordant loss/AP directions.
- Otherwise R0 is a scientific STOP.

All raw alpha rows, per-class rows, per-image quadrant counts, action rates,
parity records, provenance hashes, and gate values are written before the
decision is committed. Missing results remain missing; no interpolation is
allowed.
