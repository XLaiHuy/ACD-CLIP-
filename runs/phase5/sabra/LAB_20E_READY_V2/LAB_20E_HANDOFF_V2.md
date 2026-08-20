# LAB_20E_HANDOFF_V2

This is the final corrective source package for transfer to the lab GPU. V1 remains preserved at `runs/phase5/sabra/LAB_20E_READY/`; V2 corrects only post-training checkpoint wiring, sealed Medical execution, and bootstrap portability.

Lineage: PRE_EXTERNAL_BOUNDARY_SHA `5cacf3abc5df271adbe655c318d516777fd8ad61`; MVTEC_RESULT_SHA `b74483d6f0c3502e046e2c17ccf847d5b2bc7495`; CERTIFIED_FAST_SHA `1b4396d2fc30c61707f7ef534b4fe8b69c3b27fe`; V1 lab package `e3726014594bf90c16d389a446e9f084c4d07a19`; corrective source begins from that V1 SHA.

Scientific state is unchanged: `M1_E_Credibility`, features `E, peer_coherence, query_support_mean, peer_eigen_entropy, stage_query_profile_disagreement`, PCRR `DROP`, historical Trust-v2 external `SUPPORTED`, Authority-v2 `UNRESOLVED_MISSING_FROZEN_M0_E_CALIBRATOR`, FAST certified PASS.

Training remains VisA-only, native P1-v8.3, 20 epochs, FP32, with trainable image/text adapters and H6 Progress-1 modules; the CLIP backbone, canonical soft prompt, and fixed rho remain frozen. Training uses no Trust-v2 geometry. Checkpoint/resume state is atomic and includes optimizer, scheduler, scaler when used, all RNG states, dataloader generator, config/provenance, and checkpoint identity.

Post-training evaluators require the fixed `adapter_20.pth`. VisA uses `test.py`; MVTec uses the checkpoint-aware Trust-v2 evaluator with `--backend fast` and the unchanged GT firewall. Medical is `SEALED_MANIFEST_REQUIRED_AFTER_FINAL_CHECKPOINT_FREEZE` and requires both the existing run-local manifest protocol and `--allow-medical-evaluation`. Current Medical reads are zero.

`SOURCE_READY_FOR_LAB=true`, `FULL_20E_TRAIN_AUTHORIZED=false` (pending explicit lab authorization), `TRAINING_STARTED=false`, and `MVTEC_RERUN=false`.
