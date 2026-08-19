# TRUST_V2_M4_RECOVERY_V2

Status: frozen narrow recovery protocol, not a redesign.

## Purpose and invalid evidence

This recovery exists only to rerun the already-frozen Trust-v2/M4 analysis with collision-safe result paths. The prior Trust-v2 run is invalid because the preregistered M4 diagnostic was omitted. The subsequent corrected run is also invalid because its result paths collided with earlier untracked invalid artifacts. The immediately preceding invalid untracked versions were overwritten before preservation; because no verified backup is known, their exact previous bytes cannot be claimed recoverable. No bytes are fabricated or reconstructed. The preserved current invalid files are archived at `runs/phase5/sabra/INVALIDATED_TRUST_V2_RUNS/20260820_ARTIFACT_COLLISION/` and MUST NOT be used as evidence.

The earlier patch-specific reserve reporting defect was classified and repaired as a reporting-only defect. Neither that history nor any invalid numerical result may modify a scientific definition, threshold, feature, split, model, gate, or conclusion. This recovery is a provenance repair, not evidence that the finalized GT-free cache or corrected M4 implementation is wrong.

## Frozen scientific definitions

- M0 is `E`.
- M1 is `E + peer_coherence + query_support_mean + peer_eigen_entropy + stage_query_profile_disagreement`.
- M2 is M1 plus `S9 + R9`.
- M3 is M2 plus `S16 + R16`, and remains eligible only under the existing frozen p16 coverage rule.
- Non-PCRR model selection uses the exact previously frozen deterministic rule.
- M4 is literally the selected non-PCRR model columns in their existing order plus cached `D_rel`.
- `D_rel = abs(PGM_baseline_rank - PCRR_baseline_rank)`.
- PCRR is diagnostic-only. Its primary comparison is per held-out VisA class: `AUROC(M4) - AUROC(selected_non_PCRR_model)`. It is retained only for positive mean incremental effect, non-negative median, no catastrophic tail, and at least WEAK_POSITIVE_EVIDENCE; otherwise `PCRR_STATUS = DROP`.

No E+PCRR model, weighted fusion, PCRR rescue model, PCRR threshold tuning, feature search, or post-hoc gate change is permitted. LOCO folds, preprocessing, logistic configuration, p9/p16 geometry, PGM, Need C1, Authority-v2, evidence ladders, safety metrics, and all firewalls remain unchanged.

## Cache and provenance

The finalized GT-free cache is reused read-only from `runs/phase5/sabra/TRUST_V2_DEVELOPMENT/cache` with its existing manifest. It will not be rebuilt or modified. `D_rel` must be read from cache and must equal the cache identity `abs(baseline_pgm - baseline_pcrr)`. VisA GT may be opened only after the cache and pushed implementation gate passes.

The readiness audit records exact hashes for the manifest, every cache shard, numerical code, cache builder, audit implementation, recovery runner, checkpoint, config, CLIP, and VisA metadata.

## Collision-safe output contract

All valid recovery results MUST be written only under `runs/phase5/sabra/TRUST_V2_M4_RECOVERY_V2/`. No result may be written to `TRUST_V2_DEVELOPMENT/` or `TRUST_V2_M4_FOLLOWUP/`. Before the run, the recovery runner checks every planned result path and stops with `ARTIFACT_PATH_COLLISION` if any result or temporary output already exists. Result serialization uses temporary files followed by atomic rename and refuses overwrite. Exact-path staging is mandatory; `git add .` is forbidden.

## Workflow and stop rules

1. Freeze and push this protocol, recovery runner, readiness audit, and deterministic tests.
2. Verify local and remote synchronization and cache provenance.
3. Rerun only M0, M1, M2, M3, deterministic non-PCRR selection, corrected M4, PCRR decision, stable-but-wrong reporting, Authority-v2, statistics, adversarial review, and final VisA decision.
4. Validate every result path, cache hash, OOF isolation, literal M4 feature order, and zero MVTec/medical reads.
5. If ineligible, do not access MVTec; write the final handoff and stop. If eligible, commit and push VisA results, then freeze and push the candidate in a separate commit.
6. Only after the freeze commit is remotely verified may MVTec be accessed. Medical data is forbidden in all cases.

Any remote race, cache mismatch, output collision, GT leakage, MVTec-before-freeze access, medical access, or semantic result-changing bug stops the recovery.
