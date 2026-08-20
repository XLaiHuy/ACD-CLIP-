Lab-ready Recovery-v2 20e package

Terminal state: LAB_20E_SOURCE_READY. TRAINING_STARTED = false.
Medical reads are zero and Medical is sealed until final checkpoint.

Scientific lineage:
- candidate freeze: c878ca10c0596b7ed01909cf65de486b21946dbd
- PRE_EXTERNAL_BOUNDARY_SHA: 5cacf3abc5df271adbe655c318d516777fd8ad61
- MVTEC_RESULT_SHA: b74483d6f0c3502e046e2c17ccf847d5b2bc7495
- previous FAST/source SHA: 1b4396d2fc30c61707f7ef534b4fe8b69c3b27fe
- selected model: M1_E_Credibility
- features: E, peer_coherence, query_support_mean, peer_eigen_entropy, stage_query_profile_disagreement
- PCRR: DROP
- Trust-v2 MVTec: SUPPORTED
- Authority-v2: UNRESOLVED_MISSING_FROZEN_M0_E_CALIBRATOR
- scientific validity: VALID_POSITIVE_EXTERNAL_RESULT

Training:
The canonical P1-v8.3 entrypoint is train.py, wrapped by
tools/sabra/lab_train.py. It uses the existing VisA train loader, 20 epochs,
image size 518, batch 1, accumulation 6, FP32, gradient checkpointing, seed 0,
and the frozen H6/DFG arguments in TRAIN20E_FINAL_CONFIG.json.
The CLIP backbone and canonical H6 soft prompt are frozen. Image adapter, text
adapter, and H6 Progress-1 trainable modules are optimized; rho is fixed.
Training has no Trust-v2 geometry call. Certified FAST is the post-training
Trust-v2 evaluator backend; exact remains available as fallback.

Checkpoints are atomic each epoch: checkpoints/adapter_<epoch>.pth,
checkpoints/LATEST_CHECKPOINT.json, and final checkpoints/adapter_20.pth.
Resume restores model, optimizer, scheduler, scaler-if-used, and Python/NumPy/
torch CPU/CUDA/dataloader RNG state.

Dataset roles:
- VisA: training and explicit post-training held-out-class validation evidence.
- MVTec: previously observed external validation/frozen benchmark only.
- Medical: clean final external evaluation only, sealed by default, reads 0.

Authorization:
FULL_20E_TRAIN_AUTHORIZED = false is preserved from the frozen protocol.
SOURCE_READY_FOR_LAB = true, but this package does not authorize launching 20e.
