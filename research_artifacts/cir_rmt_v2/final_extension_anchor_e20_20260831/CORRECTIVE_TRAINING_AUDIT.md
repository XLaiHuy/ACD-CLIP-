# Corrective training audit

Status: PASS for the E14-to-E20 image-parameter-anchor continuation.

The run resumed the existing epoch-14 `last.pth` cursor; it did not restart training. It used the frozen CIR_DFG_RMT_V2 config, VisA seed 0, FP32, effective batch 6, Adam betas (0.9, 0.999), eps 1e-8, weight decay 0, gradient clipping 1.0, the existing StepLR(step_size=1, gamma=0.9), and the existing loss `cls + seg + 0.001*kg + 0.0*k`. The image-only anchor was lambda 0.001 against the frozen Phase2B E14 image adapter and was train-only.

E15-E20 telemetry, E16/E18/E20 first-batch gradient probes, exact scheduler state, optimizer group state, and checkpoint hashes are recorded in the companion JSON/CSV artifacts.

The one launch device-string failure and one resource-preflight bookkeeping failure were engineering-only, occurred before scientific work, and are recorded as resolved in FAILURE_CLASSIFICATION.json. No target cell was generated before the target-blind freeze.

Target tuning: NO. MVTec: NOT_RUN.
