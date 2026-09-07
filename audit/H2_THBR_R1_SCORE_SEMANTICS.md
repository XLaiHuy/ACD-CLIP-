# H2 THBR R1 score semantics

The canonical training segmentation tensor is `seg_pred` returned by `ACDCLIP.vision_text_fusion_gate_seg` in `train.py` with `test_mode=False`, `cir_training=False`, and equal stage fusion. It has shape `[B, 2, 518, 518]` and is a post-fusion softmax probability map produced from bilinearly resized native 37x37 stage logits. The THBR anomaly score is `seg_pred[:, 1, :, :]`, the abnormal-class probability.

The existing focal loss consumes the full two-channel `seg_pred` probability tensor directly; it does not apply another softmax. Normal Dice consumes `seg_pred[:, 0, :, :]` against `1-mask`, and abnormal Dice consumes `seg_pred[:, 1, :, :]` against `mask`. The existing task segmentation loss is focal + normal Dice + abnormal Dice.

For the frozen endpoint audit, the repository's existing evaluator semantics are preserved: each native stage score is formed from the same vision/text features, Gaussian-blurred with the existing 7x7 industrial kernel, bilinearly resized to 518x518, equally fused in logit space, and softmaxed; endpoint THBR diagnostics therefore use the same post-fusion abnormal probability score space as the bounded diagnostics. No separate score head is introduced.
