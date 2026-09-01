# Exact H2 contract for the matched master experiment

Status: CONFIRMED from historical source commit `e03966997d4cecfd985943a4053a93e1e40197ec`, the H2 run/checkpoint, and the historical evaluator. The authoritative recovered contract is also preserved in `phase2b_historical_gain_forensics_20260901/HISTORICAL_PHASE2B_CONTRACT.json`.

R uses the historical hybrid Phase2B path: ViT-L-14-336 at 518px, three groups, image/text adaptation 0.2, LoRA rank/alpha 16/2, convolutional LoRA rank/alpha 8/2 with kernels 3/5, attention DFG dimension 256 and tau 8, SS2D weight-residual beta warmup, hybrid alpha schedule 0/0.05/0.1/0.2, lambda_kg=0.01, exact detached-W_K lambda_k=0.002, AMP/autocast/GradScaler, batch/effective batch 6, Adam groups text/image/soft with base LR 5e-4/1e-3/5e-5, zero weight decay, and StepLR gamma 0.9 stepped after each epoch and before candidate save.

The recovered K-reg is the historical cosine distance in detached W_K space, mean over stages, groups, and normal/abnormal channels. The implementation test confirms nonzero K-reg and the expected soft-prompt gradient path.

The three arms share one exact H2 E0 initialization. RA adds only the frozen normalized image-adapter parameter anchor at lambda 1e-3. RCA adds only the frozen CIR-V2 train-time peer transport on top of RA. All deployment/evaluation uses native H2 alpha=0.

Historical AMP is intentionally preserved. The extension does not change precision, optimizer, scheduler, loss, augmentation, batch geometry, DFG math, scoring, or evaluator protocol.
