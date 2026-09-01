# Lost-gain decomposition

The H2 historical replay is confirmed: the exact E10 model-state checkpoint replays to the historical logged macro 90.98 AUROC / 40.35 AP. The same checkpoint under the current evaluator gives 90.9222 / 40.3731, a cross-evaluator shift of only -0.0528 / +0.0247 percentage points. Evaluator migration therefore explains only a rounding-level fraction of the H2-to-C2 E10 loss.

C2 P E10 under the current evaluator is 87.9118 / 31.8093. Relative to H2’s same-current-evaluator replay (90.9222 / 40.3731), the residual is -3.0104 / -8.5637 percentage points. This residual is not a pure training-code term: it includes the removed K-reg, lower KG weight, AMP-to-FP32 migration, prompt LR change, horizon/selection differences, any loader/augmentation migration, and the C2 E10 full-checkpoint metadata incident. It is correctly labeled a multiple-factor checkpoint/training/config component.

The scheduler is not in the lost-gain component for H2 to C2: both historical H2 and corrected C2 actually step StepLR after each epoch before saving. The missing scheduler.step remains a confirmed and major bug in the old CIR-V2 run, but it is a separate CIR-versus-parent confound and cannot explain this H2-to-C2 decomposition.

No single factor is causally isolated by the existing evidence. K-reg is associated with the H2 champion and absent in C2, but KREG_CAUSAL_STATUS=ASSOCIATED_ONLY until an exact restored-H2 comparison is run.
