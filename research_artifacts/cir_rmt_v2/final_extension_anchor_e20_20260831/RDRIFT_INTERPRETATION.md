# Representation-drift / target association interpretation

The association table joins source-only same-epoch representation drift with target-domain deltas. It is diagnostic and correlational only: source drift is repeated across domains within an epoch, and six epoch points are insufficient to establish a causal relationship.

The representation closure was classified `PRESERVATION_PARTIAL`: the image-anchor parameters were closer to P at same epochs, while only a subset of non-text feature signals showed lower drift.

| comparison | target delta metric | correlation across epochs | n epochs |
|---|---|---:|---:|
| C_OLD | anchor_train_effect_pixel_auroc | -0.738078 | 6 |
| C_OLD | anchor_train_effect_pixel_ap | +0.016796 | 6 |
| C_OLD | anchor_rmt_inference_effect_pixel_auroc | -0.698460 | 6 |
| C_OLD | anchor_rmt_inference_effect_pixel_ap | -0.702136 | 6 |
| A | anchor_train_effect_pixel_auroc | -0.341358 | 6 |
| A | anchor_train_effect_pixel_ap | +0.563557 | 6 |
| A | anchor_rmt_inference_effect_pixel_auroc | -0.242869 | 6 |
| A | anchor_rmt_inference_effect_pixel_ap | -0.143429 | 6 |

No target labels were used to tune the anchor or RMT parameters. Post-hoc GT-derived observations, if present in inherited diagnostic artifacts, remain diagnostic only.
