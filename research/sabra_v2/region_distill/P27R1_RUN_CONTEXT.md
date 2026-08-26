# P27R1 Run Context

## Identity

- Branch: `research/p27r1-cuda804-runtime-recovery-v1`
- Execution-base SHA: `de41b380449dcbc0b124f71f4f8fbb789e1a96f0`
- Recovery attempt UUID: `884f7327-1135-491b-8c12-dc188455be2c`
- Original P27 attempt UUID: `60dd4d8d-15cd-403e-b2b3-4b38f4e7da1a`
- Authoritative runner: PID 17373 when last observed; `tools.sabra_v2.run_region_distill_science`
- Cache root: `/workspace/p27r1_cache_v1`
- Run root: `/workspace/p27r1_science_v1`
- Protocol lineage: P27_REGION_DISTILL_V1; P27R1_CUDA804 recovery attempt 1.

## Scientific invariants

- P26 Phase2B and CLIP are frozen; only `RegionResidualAdapter` is trainable.
- 12-class leave-one-class-out protocol; 20 epochs, batch size 1, learning rate 0.001, seed 0.
- Frozen teacher semantics; no new scientific tuning.
- MVTec reads: 0. Medical reads: 0.
- Scoring is forbidden until all 12 immutable held predictions exist.

## Engineering recovery

- The original P27 attempt stopped because CUDA runtime propagation failed (CUDA804 class).
- P27R1 uses the prequalified host-driver-first runtime recovery. The recorded probe passed with Torch 2.5.1+cu121, CUDA available, and an RTX 3070 Ti.
- Child CUDA execution has functioned through Tier-A, Tier-B, and `candle` training/prediction without a CUDA error.
- Cache schema: `P27_CACHE_V1`; float32 NumPy memmaps. Tier-A is GT-free, class-sharded, cross-fold reusable. Tier-B is fold-local and source-inventory-only.

## Timeline / milestones

- 2026-08-25T16:48:53Z — runner started; recovery attempt recorded. Completed folds 0/12, immutable predictions 0/12, scored folds 0/12.
- 2026-08-25T17:01:03Z — Tier-A complete: 12/12 class manifests and provenance; 720.609 s. Artifact: `/workspace/p27r1_cache_v1/summaries/tier_a_build.json`.
- 2026-08-25T17:01:35Z — `candle` Tier-B complete; 28.512 s. Artifact: `/workspace/p27r1_cache_v1/summaries/tier_b_candle_build.json`.
- 2026-08-25T17:35:09Z — `candle` training complete; 2,009.712 s, 39,240 steps. Completed folds 1/12. Artifact: `/workspace/p27r1_science_v1/candle/training/TRAINING_COMPLETE.json`.
- 2026-08-25T17:35:20Z — `candle` immutable held prediction complete; 6.571 s, SHA-256 `6063037492ae31e45830b0612a15f1ff36802b68289406e92fc32e3abd7d21c9`. Immutable predictions 1/12. Artifact: `/workspace/p27r1_science_v1/candle/predictions/PREDICTION_COMPLETE.json`.
- 2026-08-25T17:35:55Z — `capsules` Tier-B complete; 31.438 s. Training began under child PID 19000. Completed folds 1/12, immutable predictions 1/12, scored folds 0/12.
- 2026-08-25T18:07:27Z — `capsules` training complete; 1,887.677 s, 40,040 steps. Completed folds 2/12. Artifact: `/workspace/p27r1_science_v1/capsules/training/TRAINING_COMPLETE.json`.
- 2026-08-25T18:07:37Z — `capsules` immutable held prediction complete; 5.302 s, SHA-256 `93b4dfafc4cef7ef6d0991e3a65b9906e73d778a5fe0d7a9b225fe3631d499f1`. Immutable predictions 2/12. Artifact: `/workspace/p27r1_science_v1/capsules/predictions/PREDICTION_COMPLETE.json`.
- 2026-08-25T18:08:11Z — `cashew` Tier-B complete; 30.639 s. Training began under child PID 19921. Completed folds 2/12, immutable predictions 2/12, scored folds 0/12.
- 2026-08-25T18:40:42Z — `cashew` training complete; 1,946.763 s, 40,240 steps. Completed folds 3/12. Artifact: `/workspace/p27r1_science_v1/cashew/training/TRAINING_COMPLETE.json`.
- 2026-08-25T18:40:52Z — `cashew` immutable held prediction complete; 5.003 s, SHA-256 `8f731d0e2c02f248a5d61ace3565860c48433a25094f979a4f802ddcc1f40ad1`. Immutable predictions 3/12. Artifact: `/workspace/p27r1_science_v1/cashew/predictions/PREDICTION_COMPLETE.json`.

## Supervision note

- A Codex/UI interruption did not interrupt scientific execution. The authoritative runner remained live at the subsequent transition check and advanced normally.
- Last known live phase: `chewinggum` Tier-B build (PID 20645 when observed). Passive supervision must wait for the authoritative child to exit, then inspect once; do not retry, alter, or score early.

## Post-run reconciliation

- 2026-08-25T23:40:55Z — all 12 immutable held predictions passed the scoring gate; the gate records exact class coverage with no duplicates.
- 2026-08-25T23:47:36Z — all 12 frozen predictions were scored exactly once with `fit_or_teacher_steps=0` for every fold.
- 2026-08-25T23:47:37Z — 12-class aggregation completed; `/workspace/p27r1_science_v1/P27R1_RUN_COMPLETE.json` recorded `COMPLETE`.
- Post-run inspection found no active runner for the previously observed PID 17373 and no restart was performed.
