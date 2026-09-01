# Final historical Phase2B root-cause audit

## Primary conclusion

H2 is confirmed and reproducible. The exact historical E10 model-state checkpoint ae27443f99020588298a9ecc6dfc833a83ebe7a752f00e8524042d5a84a2c0cb replays to the historical logged macro 90.98 / 40.35 Pixel AUROC/AP. The same checkpoint through the current evaluator gives 90.9222 / 40.3731. Current C2 P E10 is 87.9118 / 31.8093.

The same-checkpoint evaluator component is -0.0528 / +0.0247 percentage points. The residual C2-vs-H2-current component is -3.0104 / -8.5637 points. This residual is a multiple-factor H2-to-C2 migration component, not a pure training-line attribution.

## What is proven

- The historical H2 repository/run/checkpoint/evaluator identity is exact enough to reproduce the logged E10 historical result to the logged precision.
- The current evaluator is not the dominant explanation of H2’s historical gain: same-checkpoint cross-evaluator shifts are rounding-level.
- H2 actually stepped StepLR; C2 corrected parent also steps StepLR. Scheduler migration is not supported as the cause of H2-to-C2 loss.
- Current C2 K-reg is a zero stub, while H2 contains exact detached-W_K K-reg at lambda_k=0.002.
- H2 used AMP/autocast/GradScaler; C2 used FP32 without AMP. H2 used lambda_kg=0.01, soft-prompt LR 5e-5, and a 15-epoch retrospective candidate protocol; C2 differs on each.
- H2 and H1 winning checkpoint selection was retrospective Medical-informed, so those historical champions are not target-blind.

## What remains correlational or unknown

K-reg is associated with the H2 champion but its causal contribution is not isolated. Precision, KG coefficient, prompt LR, horizon, loader/augmentation, evaluator, and selection effects are jointly confounded. The C2 E10 full optimizer/scheduler/RNG payload was lost in the disclosed serialization incident; model-state metrics remain usable, but exact resume-state verification is unknown.

## Extension gate

The predecessor PA control is ingested separately. It classified CIR training as inconclusive and inference RMT as neutral. H2 is selected for restoration audit because it is the strongest same-current-evaluator pixel parent, but no restored-H2 implementation, parity test, source-only anchor gate, new Medical evaluation, or MVTec evaluation was run in this snapshot. The next authorized experiment is an exact H2-contract restoration with fixed-input parity, followed by a source-only bounded RA gate; CIR training is conditional and inference RMT remains off by default.
