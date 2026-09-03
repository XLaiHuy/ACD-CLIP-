# Seed-0 step reconciliation

Status: `PASS`.

The loader has 361 batches per epoch. Checkpoint payloads and train logs reconcile exactly with the optimizer-step rule: `global_step` advances after a successful scaler/optimizer step, while nonfinite-gradient batches are recorded and skipped.

For example, H E15 is `359 + 14 x 361 - 3 = 5410`; A E15 is `359 + 14 x 361 - 2 = 5411`; and C E15 is `359 + 14 x 361 - 3 = 5410`. AC resumed from its adapter-3 checkpoint at the E4 boundary; its E15 step is `359 + 2 x 361 + 12 x 361 - 1 = 5412`. The E20 values reconcile by adding five epochs and the logged skips.

There were no nonfinite-loss skips, duplicate batches, repeated epochs, skipped resume epochs, hidden extra steps, or target-guided retraining. The compact skip fields in the original freeze manifest are retained as historical metadata; the per-epoch log reconciliation in the JSON file is authoritative.
