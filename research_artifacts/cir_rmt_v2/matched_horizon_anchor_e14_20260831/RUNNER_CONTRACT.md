# Matched-horizon E14 speed contract

This runner is the only authorized path for the selected `SELECTIVE_PHASE2B_ANCHOR` validation at this stage.

- Train VisA CIR only through E14.
- Retain scientific candidate checkpoints only at E10, E12, and E14.
- Update `last.pth` atomically as an ephemeral resume cursor; it is not an evaluation candidate.
- Reuse the frozen P/C0 E10/E12/E14 source rows from the prior archive. The post-hoc evaluator forwards new CIR checkpoints only.
- Run representation, AP-tail, deployment, branch, and held-out diagnostics after training on the preregistered bounded source sample, never as extra training-step forwards.
- Keep the Phase2B E14 anchor resident on the selected training device for the whole process. Profile its overhead once before training; no vectorization is enabled unless exact loss/gradient parity is established.
- Run the structural E10 catastrophic-failure gate. It can stop for nonfinite state, checkpoint identity failure, severe anchor-gradient domination, or inactive RMT; it never stops for a small metric difference.
- Hold a process lock for the full runner and resume from `last.pth` after an interruption instead of restarting.
- No AMP, TF32, batch, optimizer, LR, scheduler, resolution, augmentation, evaluator, Gaussian/deployment, metric, Medical, or MVTec changes are part of this path.

Expected final measurements are written to `MATCHED_HORIZON_TIME_REPORT.md`: seconds per epoch, images/sec, peak allocated/reserved VRAM, anchor overhead, E10/E12/E14 evaluation time, and total wall time.
