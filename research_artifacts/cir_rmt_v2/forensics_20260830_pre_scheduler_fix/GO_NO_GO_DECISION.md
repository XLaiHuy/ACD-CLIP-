# CIR_DFG_RMT_V2 go/no-go decision

DECISION: `KEEP_PARENT_FIX_TRAINING`

Rationale: CIR_SCHEDULER_BUG_CONFIRMED: current CIR-V2 training is not optimization-matched to Phase2B because train_full.py never calls scheduler.step(). The present benchmark cannot cleanly isolate the RMT hypothesis; run one matched corrective retrain before attributing degradation to RMT.

Scheduler audit classification: `CIR_SCHEDULER_BUG_CONFIRMED`.
Current CIR-V2 training was not optimization-matched to Phase2B; this benchmark cannot cleanly isolate the RMT hypothesis, and the degradation must not be attributed directly to RMT before the matched corrective retrain.
Recommended next experiment: one matched parent/CIR retrain with the same seed, VisA source, CLIP asset, FP32, effective batch, Adam, StepLR timing, losses, and checkpoint schedule; only CIR/RMT differs.

Full medical alpha=0.5 minus alpha=0 summary: `{"alpha05_better_both": 8, "alpha05_better_pixel_ap": 11, "alpha05_better_pixel_auroc": 10, "alpha05_worse_both": 17, "mean_delta_pixel_ap": -0.00034197302232319193, "mean_delta_pixel_auroc": -0.00015730314557440932, "n": 30}`.

This is the sole decision for this audit. No architecture change, MVTec training, or overwrite of frozen artifacts is authorized by this file.
