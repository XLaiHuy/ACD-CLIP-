# SABRA I/O contract

The canonical pipeline is shared by `train.py`,
`select_phase2b_checkpoint.py`, `calibrate_sabra.py`, and `test.py`.

The Phase2B forward contract is:

```text
seg_features        [3, B, 1369, 768]
det_features        [3, B, 768]
text_features       [3, B, 768, 2]
native_logits       [3, B, 1369, 2]
native_margin       [3, B, 1369]
native_segmentation [B, 518, 518]
classification      [B]
```

SABRA consumes the same frozen native tensors for both comparison arms.
Trust consumes exactly `[E, peer_coherence, query_support_mean,
peer_eigen_entropy, stage_query_profile_disagreement]`. Need consumes exactly
`[margin_within_image_rank, robust_margin_normalization, D_rank,
deployment_sensitivity]`. Authority is `T*N`. Correction is
`delta=lambda*margin_scale*T*N`, shared across all three stages, zero in the
normal channel, and positive in the abnormal channel.

Relational construction is image-only and receives no masks, labels, or mask
paths. VisA GT may be joined only after relational records are frozen.
