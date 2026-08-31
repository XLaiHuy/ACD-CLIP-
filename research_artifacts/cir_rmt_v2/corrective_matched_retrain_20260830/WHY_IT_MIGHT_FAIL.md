# Why the leading future direction might fail

An anchor can over-constrain VisA adaptation, lower source localization quality, or preserve the wrong parent geometry. The source/Medical split may also reflect domain shift that parameter proximity cannot solve. The measured train/deploy operator mismatch could dominate Pixel AP, in which case a representation penalty would not address the real bottleneck. Finally, the current target matrix has no paired bootstrap intervals, so a small apparent change could be noise.

Therefore the proposal must be falsified source-only, one variable at a time, with Pixel AUROC, Pixel AP, image metrics, checkpoint stability, and the frozen deployment path. Do not select its coefficient from Medical targets.
