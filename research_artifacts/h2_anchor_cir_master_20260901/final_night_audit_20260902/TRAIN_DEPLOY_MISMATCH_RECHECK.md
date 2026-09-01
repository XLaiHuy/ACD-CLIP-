# Train/deploy mismatch recheck

Status: `SECONDARY`; audit only, no operator change.

Training segmentation uses native interpolation, stage aggregation, and softmax. Deployment uses the H2 evaluator path, which adds Gaussian smoothing before interpolation, stage aggregation, and softmax. The H2 adapter source contains this branch under `test_mode` and the exact Medical evaluator uses `test_mode=True`.

The inherited deterministic first-batch V2 measurement found nonzero train/deploy divergence: mean absolute map difference approximately `0.00401827`, maximum absolute difference approximately `0.998923`, and Pearson correlation approximately `0.63297`; the corresponding example train/deploy Pixel AUROC/AP values were `0.997606 / 0.685499` versus `0.995797 / 0.553641`.

The same H2 train/deploy operator is used for historical H2, R, and RA. No new per-arm operator comparison was run tonight. Because the operator is common to the arms, it cannot by itself explain the RA-versus-R same-E10 contrast. Its effect on absolute Medical metrics remains correlational and is retained as a protocol risk for future reporting.
