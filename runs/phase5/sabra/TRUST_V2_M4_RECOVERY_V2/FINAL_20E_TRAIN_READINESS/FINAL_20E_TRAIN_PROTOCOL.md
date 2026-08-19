# Final 20e Training Readiness — SABRA Recovery-v2

Status: persisted readiness record; execution is not authorized.

This package is created only after the Trust-v2 M4 Recovery-v2 scientific
pipeline reached its valid terminal state. It records the actual outcome and
is not a new scientific protocol, a redesign, or permission to begin a
training run.

## Scientific inheritance

- Recovery-v2 VisA candidate: `M1_E_Credibility`.
- Frozen feature order: `E`, `peer_coherence`, `query_support_mean`,
  `peer_eigen_entropy`, `stage_query_profile_disagreement`.
- M4 diagnostic: the selected non-PCRR columns in that exact order plus
  `D_rel = abs(PGM_baseline_rank - PCRR_baseline_rank)`.
- PCRR status: `DROP`.
- Trust-v2 status: `SUPPORTED`.
- Need C1 status: `SUPPORTED`.
- Authority-v2 status: `SUPPORTED`.
- VisA candidate eligibility: passed and candidate freeze was pushed.
- MVTec external validation: unavailable because the required dataset image
  root was not present. No external metrics are claimed.

## Authorization state

`FULL_20E_TRAIN_AUTHORIZED=false`.

The false state is inherited from the actual terminal: MVTec external
validation could not be performed, so the next full-training study is not
scientifically authorized by this package. `EXPLORATORY_20E=false` and no
training has been started.

This package must not be used to bypass frozen Trust-v2 gates, fabricate
MVTec evidence, alter thresholds, retune components, redesign Phase2B, or
access medical data. Any future non-authorized experiment must be separately
labelled `EXPLORATORY_20E=true` and must not be represented as an authorized
scientific continuation.

## Reproduction boundary

The exact cache, model checkpoint, CLIP asset, configuration, source hashes,
protocols, result artifacts, invalidation provenance, and resume commands are
listed in `FINAL_20E_ARTIFACT_MANIFEST.json` and
`FINAL_20E_RESUME_COMMANDS.md`. Large required files are stored with exact
Git LFS path rules and must be fetched before reproduction.

No Phase2B inference/cache generation, MVTec access, medical access, or 20e
training is authorized by this readiness record.
