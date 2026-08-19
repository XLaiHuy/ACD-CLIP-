# SABRA Trust-v2 Recovery-v2 handoff

The Recovery-v2 scientific pipeline completed through VisA candidate freeze. The preceding artifact collision was preserved at `runs/phase5/sabra/INVALIDATED_TRUST_V2_RUNS/20260820_ARTIFACT_COLLISION/`; its overwritten earlier untracked bytes were not reconstructed.

VisA selected `M1_E_Credibility` with features `E, peer_coherence, query_support_mean, peer_eigen_entropy, stage_query_profile_disagreement`. Corrected M4 is the selected non-PCRR model plus cached `D_rel = abs(PGM_baseline_rank - PCRR_baseline_rank)`. PCRR status is `DROP`. Trust-v2, Need C1, and Authority-v2 are `SUPPORTED`; p16 coverage is `STRONG`.

The candidate freeze was pushed and verified. MVTec external validation was authorized only after that freeze, but no MVTec image root was available. No MVTec image or mask reads, metrics, or confirmatory evidence are claimed. MVTec image/mask reads are zero, the metadata probe count is one, and medical reads are zero.

The final readiness package is persisted at `runs/phase5/sabra/TRUST_V2_M4_RECOVERY_V2/FINAL_20E_TRAIN_READINESS/`. Its exact persistence boundary is readiness commit `a1de69cf0d365ab34e1ac6257453806e7003a22d`. `FULL_20E_TRAIN_AUTHORIZED=false`, `EXPLORATORY_20E=false`, and no training has started. This package is non-executable and does not authorize medical access, Phase2B redesign, retuning, or an exploratory run without explicit exploratory labelling.

The final handoff update is the repository boundary immediately following the readiness commit. Do not rerun Phase2B, rebuild the GT-free cache, alter formulas, retune thresholds, change gates, or use invalid archived artifacts as evidence.

## Terminal external-validation failure

The MVTec external-validation stage is preserved as `EXTERNAL_VALIDATION_FAILURE` with `VALID=false` at `runs/phase5/sabra/TRUST_V2_M4_RECOVERY_V2/FAILED_EXTERNAL_VALIDATION_20260819_MVTEC_UNAVAILABLE/`. All audited image-root locations were absent; MVTec image/mask reads are zero, metadata probe count is one, and no MVTec metrics are claimed. The VisA result and GT-free cache remain valid and usable. Full-20e training remains unauthorized.
