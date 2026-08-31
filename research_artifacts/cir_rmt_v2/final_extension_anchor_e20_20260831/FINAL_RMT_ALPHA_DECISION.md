# Final RMT alpha decision

The inference comparison is A05 - A0 at the same anchored checkpoint. It is conditional on the anchored representation and is not a training comparison.

| metric | mean A05-A0 over domain/epoch | positive cells | total cells | sign fraction |
|---|---:|---:|---:|---:|
| pixel_auroc | -0.00008757 | 2 | 36 | 0.056 |
| pixel_ap | -0.00010050 | 2 | 36 | 0.056 |
| image_auroc | +0.00000807 | 12 | 18 | 0.667 |
| image_ap | +0.00000610 | 13 | 18 | 0.722 |

Interpretation is descriptive and uses no target-domain hyperparameter selection.

Overall experiment decision: `KEEP_ANCHOR_DISABLE_INFERENCE_RMT_CANDIDATE`.
