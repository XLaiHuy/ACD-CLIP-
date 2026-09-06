# H2 historical mixed FP16+FP32 V1 smoke

SMOKE_TEST=PASS
CHECKPOINT_RESUME_SMOKE=PASS
DTYPE_TRACE=PASS

- Scope: source-only VisA; 30 attempted batches total (5 fresh E1 + 5 each E2-E6); no target evaluation.
- Fresh/save/reload/resume: PASS; E1 global step 5, E6 global step 30.
- Checkpoint v3 state: model, optimizer, scheduler, GradScaler, Python/NumPy/Torch/CUDA RNG, and dataloader generator present.
- RNG/dataloader identity: PASS; saved states reproduce byte-for-byte across independent reloads; 25 resumed augmented-batch identities logged.
- Scientific identity: HISTORICAL_MIXED_FP16_FP32_V1, FP16 autocast, GradScaler on, TF32 off, later transformer islands off.
- FP32 persistence: model and optimizer floating state are FP32.
- Anchor: active from E2 with lambda 0.0021633926715180626 and family cap 0.1.
- Trainables/LR: exact counts 14,095,887 while prompt-frozen and 14,102,031 after unfreeze; image/text StepLR and constant active prompt LR verified per epoch.
- Maximum effective active-family Anchor ratio: 0.0999995974 (cap 0.10).
- DFG/SS2D: active; FP32 weight residual; beta schedule observed E2-E3=0 and E4-E6=0.05.
- Prompt: hybrid; alpha schedule observed E2-E3=0, E4=0.05, E5=0.10, E6=0.20.
- Numerical events in smoke: nonfinite loss skips 0; recoverable gradient skips 0. The smoke is not required to reproduce the historical two skips.
- E1 checkpoint SHA256: `7d0240dac5a8518514afe2af8f765a645787cc529f6e7c79747972f19356e7a8`
- E6 checkpoint SHA256: `c18d8785998af89f699c76c88a0360a35b2fd6b0b0d104a7abbb2dbb223270bd`
