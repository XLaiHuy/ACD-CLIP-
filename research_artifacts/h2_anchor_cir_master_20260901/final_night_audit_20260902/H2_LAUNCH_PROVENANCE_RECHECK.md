# H2 launch provenance recheck

Status: `PASS` for the requested launch facts. The actual `lambda_k` and training horizon are proven from the historical `train.log`, the matching repository launcher, and checkpoint metadata.

Evidence priority was applied in this order: checkpoint metadata, exact log args/state lines, matching Git launcher, then directory naming. The historical log records `lambda_k=0.002`, `lambda_kg=0.01`, `image_lr=0.001`, `text_lr=0.0005`, `soft_prompt_lr=0.00005`, `lr_gamma=0.9`, `batch_size=6`, `amp=True`, `num_workers=6`, `grad_clip_norm=1.0`, and `non_finite_loss_abort_threshold=20`. The historical adapter payloads independently retain `lambda_k=0.002` and `lambda_kg=0.01`.

The historical run contains `adapter_1.pth` through `adapter_15.pth`, and the log records a 15-epoch run. Therefore:

`H2_ACTUAL_LAMBDA_K=0.002`

`H2_ACTUAL_HORIZON=E1-E15`

The historical `train.py` constructs `torch.optim.Adam` with default betas `(0.9, 0.999)`, `eps=1e-8`, and `weight_decay=0`, with groups ordered text, image, soft prompt. It constructs `StepLR(step_size=1, gamma=0.9)`, calls `scheduler.step()` after the epoch loop, reapplies the constant soft-prompt policy, and only then writes the model-only checkpoint. The soft prompt is frozen through E3 and uses `5e-5` after unfreeze. AMP/GradScaler is enabled by the launcher.

The H2 master manifests independently show corrected post-epoch LR histories. For example, current R E10 has image LR `0.0003486784401`, text LR `0.00017433922005`, prompt LR `0.00005`, scheduler `last_epoch=10`, and `_step_count=11`.

Limits are important. The historical run does not preserve a run-local launcher, a seed, `PYTHONHASHSEED`, optimizer/scheduler state, or RNG state. The matching repository launcher and the exact log args were therefore used to reconstruct provenance; this proves the hyperparameters and horizon, not exact stochastic replay. See `H2_TRAINING_REPRO_GAP.md`.
