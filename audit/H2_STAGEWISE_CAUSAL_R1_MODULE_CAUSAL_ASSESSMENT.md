# H2 Stagewise Causal Localization Audit R1 — Module Causal Assessment

## Frozen diagnosis context

The endpoint oracle supports final AP/AUROC improvement from a local Stage-2
replacement on both frozen VisA cohorts, and matched far-background
replacement does not reproduce that endpoint benefit. However, the required
local-ranking gate is not satisfied on both cohorts: Stage 2 does not beat both
Stage 1 and Stage 3 on both anomaly-vs-near AP and AUROC. Therefore
`STAGE2_UNIQUE_CAUSAL_SUPPORT=NO`, and Stage 2 is not an established sole root
cause. Exact patch occupancy also shows elevated zero-footprint near scores
and large partial-footprint scores.
The deterministic trajectory replay is invalid for causal trajectory claims:
the control replay is exact, but the candidate replay differs from the
committed candidate endpoint by max absolute model-state difference 0.1016087.

## Directional evidence

The fixed 16-batch no-step preflight produced the raw S2-LOCR direction. On a
single fixed active anomalous probe item, unit descent along that direction was
used to measure directional derivatives of Stage-2 near/interior/boundary/
positive/far margins, fused margins, S1/S3 margins, and the main task loss.
The strongest LOCR gradient norms were `seg_proj=17.5977`,
`lora_adapters=8.3420`, and `m_i_w=5.7875`. Conv-LoRA produced the largest
cross-stage derivative in the probe (`S3 near=-10.36`) and increased the main
task loss along LOCR descent (`+1.8788`); `m_i_w` also increased task loss
(`+0.4205`). These are directional diagnostics, not update results.

## Module evidence ledger

* **Conv-LoRA — `SUPPORTED`.** It has a large raw LOCR direction, suppresses
  Stage-2 near/interior/positive margins, changes the fused near margin, and
  has a strong Stage-3 directional effect. This supports a coupled parameter
  pathway, not the claim that Conv-LoRA alone is the unique root cause.
* **Segmentation projection — `SUPPORTED`.** It has the largest raw LOCR
  direction and suppresses Stage-2 near/interior/positive margins. Its S1/S3
  effects are small on the fixed probe, so its evidence is primarily local
  Stage-2 suppression rather than cross-stage transport.
* **DFG Q/K attention — `WEAK`.** Q/K have nonzero directional derivatives,
  generally suppressing Stage-2 regions, but their raw directions are much
  smaller than Conv-LoRA, `m_i_w`, and `seg_proj`. The committed historical
  preflight described DFG routing as healthy with low instantaneous Q/K
  gradients. No attention bypass or attention-specific training was run.
* **SS2D — `WEAK`.** The SS2D branch direction is nonzero but tiny
  (`5.5756e-6` aggregate norm) and has no distinctive cross-stage signature on
  the fixed probe; the raw-gamma family is inactive. Historical SS2D routing
  was healthy with low instantaneous gradients. No SS2D bypass training was
  run.

## Claim discipline

Facts are the exact cohort manifests, endpoint oracle metrics, occupancy
statistics, committed R1 red-team deltas, and no-step directional derivatives.
The causal conclusion supported by those facts is that Stage 2 is an
informative intervention site, while the dominant no-step directional
pathways are coupled suppression through segmentation projection / Conv-LoRA /
image-side mixing, with Conv-LoRA visibly transporting effect into Stage 3.
Patch
footprint aliasing is a plausible co-mechanism because zero-footprint near
scores exceed zero-footprint far scores and partial-footprint scores are much
larger. Unsupported claims are that attention or SS2D alone is the root cause,
that the invalid replay establishes an update-time trajectory mechanism, or
that any new mechanism is implementation-authorized.

`NEW_MECHANISM_TRAINING_RUN=NO`; `IMPLEMENTATION_AUTHORIZED=NO`.
