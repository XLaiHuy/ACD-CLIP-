# Recovery-v2 PRE-20E handoff

Scientific lineage:
- STARTING_HEAD: b74483d6f0c3502e046e2c17ccf847d5b2bc7495
- PRE_EXTERNAL_BOUNDARY_SHA: 5cacf3abc5df271adbe655c318d516777fd8ad61
- MVTEC_RESULT_SHA: b74483d6f0c3502e046e2c17ccf847d5b2bc7495

Candidate:
- selected model: M1_E_Credibility
- feature order: E, peer_coherence, query_support_mean, peer_eigen_entropy, stage_query_profile_disagreement
- PCRR: DROP

MVTec:
- source/root: /workspace/data/mvtec_ad
- exact result: runs/phase5/sabra/TRUST_V2_EXTERNAL_MVTEC_20260820_RESTART_5cacf3a
- Trust-v2: SUPPORTED (mean delta 0.21074739126824538, median delta 0.17836257309941517, positive 14/15)
- Authority-v2 primary: UNRESOLVED_MISSING_FROZEN_M0_E_CALIBRATOR
- Authority-v2 secondary raw diagnostic: FALSIFIED
- scientific validity: VALID_POSITIVE_EXTERNAL_RESULT
- medical reads: 0

FAST:
- exact reference: /workspace/ACD-CLIP-exact-ref at b74483d6f0c3502e046e2c17ccf847d5b2bc7495
- implementation: tools/sabra/trust_v2/fast_geometry.py
- certification: PASS
- discrete parity: PASS (0 mismatches)
- max c error: 0.0
- max G error: 0.0
- max eig/PGM error: 1.9073486328125e-06
- max mapped-score error: 0.0
- AUROC/AP deltas: 0.0 / 0.0
- exact runtime: 3.661146771046333 s (backend benchmark, 3 complete images)
- FAST runtime: 1.6854537070030347 s (backend benchmark, 3 complete images)
- speedup: 2.172202508935323x
- peak VRAM: allocated 2516587008, reserved 4303355904 bytes

Training:
- FULL_20E_TRAIN_AUTHORIZED: false
- training_started: false
- PRE_20E_SETUP_REQUIRED: true

Next legal action: separate 20e setup/audit prompt.
