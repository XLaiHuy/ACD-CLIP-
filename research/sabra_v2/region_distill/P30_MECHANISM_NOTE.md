# P30 Mechanism Note — Directional Distillation

This note is the implementation trace used to design P30. It is based on the
actual cached P29 scientific path, not the older uncached prototype entrypoint.

## 1. P29 objectives affecting the student

`run_p29_sign_guarded_science.py` invokes
`train_region_distill_p29_cached.py`. The cached trainer computes
`p29_sign_guarded_loss`, whose total is the unweighted sum:

```text
L_P29 = L_value + L_sign + L_normal
```

`L_value` is normalized SmoothL1 transfer, `L_sign` penalizes an oppositely
signed student residual using the absolute teacher residual, and `L_normal`
penalizes positive student residuals in source pure-normal regions. All three
terms backpropagate through the same `RegionResidualAdapter` output.

## 2. Where conflict can occur

The value term asks the student to reproduce teacher magnitude. The sign term
has a different piecewise directional gradient, and the normal term suppresses
positive corrections in a mask-derived subset even when the teacher has useful
positive activity there. Their gradients are summed before the single adapter
update. The forensic also measured a strongly opposing diagnostic segmentation
anchor gradient (`cos(g_seg, g_P29_value) ≈ -0.99996` at zero initialization),
although that anchor is not part of the cached P29 scientific loss.

## 3. Likely starvation mechanism

P29R1 measured zero-init `g_sign = 0` and `g_normal = 0`, while `g_value` was
non-zero and only `0.2876` of the same-batch raw P27-distillation gradient.
Therefore the auxiliary sign/normal terms are the most likely source of a
weak, piecewise, competing signal when they become active; the evidence does
not justify blaming either auxiliary term in isolation. P30 removes both
auxiliary losses so the directional signal can be tested cleanly.

## 4. Teacher correction tensor

The cached `tier_b/<held-class>/teacher_region.npy` tensor is `[B, 9, 9]`.
It is produced by `build_source_teacher_region_target` from frozen native
logits and source-only masks: historical R0 signed actions are converted into
a scalar patch correction and adaptively average-pooled from `37×37` to
`9×9`. In the P29 loss it is broadcast over the three stages to
`[3, B, 9, 9]`.

## 5. Student correction tensor

`RegionResidualAdapter(seg_features)` produces the student correction
`[3, B, 9, 9]`. `forward_region_student` upsamples it to `37×37` and applies
the unchanged symmetric margin integration to frozen native logits. P30 uses
the same adapter and deployment path.

## 6. Direction dimensions

The forensic compares staged teacher and student residuals elementwise over
three stages and the `9×9` region grid. P30 therefore defines one direction
vector per sample by flattening `[3, 9, 9]` into `243` signed correction
coordinates. The teacher is the same `[9×9]` direction repeated for each
stage, preserving the existing P29 semantics.

## 7. Is magnitude necessary?

Exact magnitude equality is not necessary for the P30 hypothesis. P29R1 found
that P29's mean residual magnitude was already `0.8723` of P27 while sign
agreement still slightly declined and AUROC regressed. P30 consequently
normalizes each sample's correction vector before comparing it; magnitude
behavior is measured as a safety outcome rather than optimized directly.

## 8. Can teacher signals be cached?

Yes. The existing Tier-B `teacher_region.npy` is source-only, immutable,
float32, identity-checked, and already reused by P29. Tier-A frozen features
and native logits are also reused. P30 adds no cache build and no teacher
forward.

## 9. Inference-time cost

The adapter architecture, symmetric margin integration, P26 deployment, and
prediction path remain unchanged. Directional normalization is training-loss
only, so the intended P30 inference overhead is zero percent.

## 10. Smallest hypothesis test

Keep the P29 adapter, cached Tier-A/Tier-B inputs, LOCO folds, optimizer,
schedule, FP32 policy, and evaluation semantics. Replace only the P29 loss
with one fixed-epsilon, per-sample cosine directional loss over the flattened
`[3,9,9]` correction vector. Exclude an exactly zero teacher vector from the
directional average, return an exact graph-connected zero when no valid target
exists, and add no magnitude, sign, ranking, feature, segmentation, or
calibration penalty. This gives the zero-initialized student a finite,
non-zero directional gradient while directly testing removal of mixed
objectives.
