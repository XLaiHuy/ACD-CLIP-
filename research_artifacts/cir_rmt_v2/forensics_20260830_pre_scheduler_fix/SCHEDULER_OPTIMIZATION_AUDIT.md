# CIR versus Phase2B LR scheduler and optimization audit

Classification: `CIR_SCHEDULER_BUG_CONFIRMED`

## Finding

`scripts/cir_rmt/train_full.py` constructs `StepLR(step_size=1, gamma=0.9)` and stores its state, but the epoch loop has no `scheduler.step()` call. The canonical `train.py` Phase2B loop calls `scheduler.step()` after the batch loop and before the epoch row and checkpoint payload are written.

All 5 epoch CIR checkpoints have a serialized scheduler state whose `last_epoch` is 0 while the checkpoint epoch is in [12, 14, 16, 18, 20]. Their optimizer group LRs therefore remain at the initial values (except any soft-prompt freeze policy state), rather than following the intended StepLR decay.

## Plausible magnitude (not a causal estimate)
At epoch start, CIR image/text LR exposure relative to the canonical parent is E12: 3.19x, E14: 3.93x, E16: 4.86x, E18: 6.00x, E20: 7.40x; the image and text ratios are identical because their initial LR ratio is fixed at 2:1. At E20, the canonical post-step checkpoint values would be image 1.2158e-4 and text 6.0788e-5, versus CIR 1e-3 and 5e-4, an approximately 8.23x excess. The soft prompt follows a separate constant-LR freeze/unfreeze policy: against the current canonical parent value 1e-4, CIR is 1.0x at E12-E20; the CSV's 2.0x at E12/E14 reflects the legacy history's 5e-5 prompt base, not StepLR.
These are exposure ratios, not a predicted percentage of the medical-score gap: Adam's adaptive moments, gradient clipping, skipped/non-finite updates, and representation trajectory make parameter displacement nonlinear. The scheduler bug can plausibly explain instability and late-epoch drift, but its causal share requires the matched corrective retrain.

## Checkpoint coverage

CIR epoch checkpoints: [12, 14, 16, 18, 20]; E10 is absent. CIR `last.pth` records: 1. Available historical Phase2B checkpoint artifacts: 1.

The available historical Phase2B `adapter_10.pth` is a legacy artifact and does not contain serialized `optimizer_state` or `scheduler_state`. The comparison table uses actual start-of-epoch LR observations from its `train.log` for E1-E15, canonical expected values where the log has no later epoch, and separate serialized-parent columns; it does not claim that the legacy checkpoint itself stores optimizer/scheduler state. The legacy log uses soft-prompt LR 5e-5, while the current canonical parent config uses 1e-4.

## LR convention

The required table includes actual serialized CIR values and any serialized parent values. It also includes protocol-expected parent start-of-epoch and post-`scheduler.step()` checkpoint values. The user-facing estimates such as E10 approximately 3.87e-4 correspond to the start-of-epoch convention; the canonical checkpoint is saved after the epoch scheduler step.

## Optimization details

- Both canonical trainers construct three named Adam groups: `image_adapter`, `text_adapter`, and `soft_prompt`.
- The CIR checkpoint group state reports Adam defaults (`betas=(0.9, 0.999)`, `eps=1e-8`) and zero weight decay for all three groups; no group is exempt from decay because decay is zero globally.
- The historical Phase2B `train.log` records image/text start-of-epoch LRs decaying by gamma=0.9 from E1 through E15, including E10=3.8742e-4, E12=3.1381e-4, and E14=2.5419e-4; this verifies actual parent-run decay even though its checkpoint omits optimizer/scheduler state.
- Both canonical loops clip gradients once per optimizer step after gradient accumulation. CIR uses `clip_grad_norm_` directly; Phase2B uses its equivalent helper.
- The soft prompt remains in the optimizer. `_set_epoch_state` sets its LR to zero through the freeze epochs and restores `constant_lr` afterward; this is a separate freeze/unfreeze policy, not evidence that StepLR was applied to the CIR run.
- Both trainers load optimizer state, scheduler state, RNG state, and resume at `checkpoint_epoch + 1`. CIR's resume path is mechanically present but semantically incorrect for the intended schedule: restoring a stale scheduler state preserves the constant-LR bug rather than repairing it.

## Scientific implication

The scheduler mismatch is a major protocol confound. The current CIR-V2 benchmark cannot cleanly isolate the RMT hypothesis from an optimization mismatch. The correct next action is one matched corrective parent/CIR training comparison with the same seed, source, FP32 policy, effective batch, optimizer, scheduler, scheduler timing, losses, and checkpoint schedule; no immediate architecture change or MVTec training follows from this audit.

## Evidence files

- `scheduler_optimization_audit.csv` — required epoch comparison table.
- `scheduler_optimizer_group_detail.csv` — every available CIR checkpoint and parent artifact group/state detail.
- `scheduler_audit_summary.json` — classification, inventory, and source-evidence summary.
