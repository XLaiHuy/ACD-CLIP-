# Scheduler forensics

## H2 historical Phase2B

The H2 source constructs StepLR(optimizer, step_size=1, gamma=0.9). The exact epoch loop calls scheduler.step once after the complete epoch, then reapplies the constant soft-prompt LR policy, then writes adapter_<epoch>.pth. The H2 log records the decayed schedule and historical optimizer group policy.

## C2 corrected parent

The corrected canonical trainer constructs the same StepLR and calls scheduler.step after the epoch and before the history row/checkpoint save. C2 candidate metadata records post-step scheduler state and decayed image/text learning rates.

## Old CIR-V2 run

The pre-fix CIR trainer constructed StepLR but did not call scheduler.step in its epoch loop. Its E12/E14/E16/E18/E20 checkpoint states retained last_epoch=0, _step_count=1, and base image/text learning rates. That bug is independently CIR_SCHEDULER_BUG_CONFIRMED in the preserved pre-fix archive and plausibly explains old CIR instability/excessive late LR exposure.

Conclusion for this audit: SCHEDULER_CAUSAL_STATUS=NOT_SUPPORTED for the H2-to-C2 lost-gain comparison, while the old CIR scheduler bug remains proven as a separate protocol failure.
