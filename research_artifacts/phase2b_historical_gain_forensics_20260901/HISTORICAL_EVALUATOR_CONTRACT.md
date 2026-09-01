# Historical/current evaluator contract

Historical H1/H2 evaluation uses the legacy test.py path. H2 source hash: 7bdd8cc6ada90467285a79ced9599ed778c6dc2a0ba6596d2f3311fa637fae9d; H1 source hash: e6c768b5604ea7c7c0dea7c3db709405da6876ff01a22ddecb2dde6a4f59334f. Both use exact torchmetrics when thresholds are unset, apply pixel_stride=4 to prediction and masks, round metrics before percentage reporting, and use the image score 0.5 classification + 0.5 max pixel score. Colon image metrics are unsupported and appear as zero in the legacy log; this archive normalizes them to undefined and excludes them from the three-target image macro.

The current evaluator implementation hash is cbcfaf4b2eda645fc6b440ed9bb486b5fb6b6f3af908e1c0ec70bafe13db0797. The same H2 model-state checkpoint was replayed under it. Pixel AUROC/AP changed by only -0.0528/+0.0247 percentage points at the six-domain macro; image AUROC/AP changed by -0.0207/+0.0107 points at the three-domain macro. These values are within the precision implied by the historical two-decimal output and do not support evaluator migration as the primary H2-to-C2 loss.

Historical and current values are separated in HISTORICAL_REPLAY_RESULTS.csv and SAME_CHECKPOINT_CROSS_EVALUATOR.csv.
