# Post-corrective research scope lock

Status: LOCKED FOR THE DIAGNOSIS SNAPSHOT ONLY

## Measurable bottleneck

The remaining gap is target-domain pixel transfer, especially Pixel AP. In
the exact six-target Medical matrix, corrected C0 is below matched P in
Pixel AUROC and Pixel AP at every candidate epoch, while Image AUROC is
often higher and Image AP is mixed. C05 does not recover the pixel gap:
its paired inference effect is effectively zero at macro level.

## Primary bottleneck

K2_SOURCE_GOOD_TARGET_GENERALIZATION_BAD with a K3_PIXEL_AUROC_GOOD_AP_BAD
phenotype. The source matrix is strong and C0 source Pixel AP is above P at
all six candidate epochs, but this source advantage does not transfer to the
Medical pixel metrics.

## Secondary bottleneck

K7_TRAIN_DEPLOY_OPERATOR_MISMATCH is a measured consistency risk, not yet a
proven causal explanation. A bounded corrected E14 training-batch audit found
mean absolute map difference 0.0032489155, maximum difference 0.9993287325,
and Pearson correlation 0.6475440882 between the training-side and deployed
probability paths.

## Explicitly out of scope for this snapshot

- no optimizer or loss rewrite;
- no scheduler change beyond the already-pushed matched-training fix;
- no RMT/SAR-RMT implementation;
- no target-specific threshold or alpha tuning;
- no MVTec training;
- no new full 20-epoch run;
- no target-GT-derived hyperparameter rule.

The next scientific change, if separately authorized, must be one variable at
a time and source-only first. The diagnosis does not authorize Stage 2
implementation in this commit.
