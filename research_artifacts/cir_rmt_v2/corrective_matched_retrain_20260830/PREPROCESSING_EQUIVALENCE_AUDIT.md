# Preprocessing and train/deploy equivalence audit

Status: COMPLETED as a code-path audit plus a bounded corrected E14 numerical diagnostic. Full prediction-level causal attribution remains unknown.

## Matched paths

Parent and CIR VisA training both instantiate `TextAndImageDataset` with the same VisA manifest, image size 518, bicubic RGB resize, `ToTensor`, CLIP normalization, mask nearest-neighbor resize, and binary masks. Both use the same random training augmentations and loader geometry in the matched retrain. The 46-row match ledger records these fields as matched.

For Medical evaluation, P, C0, and C05 all use the same `BaseSingleClassDataset` path through `_target_dataset`, with deterministic RGB bicubic resize to 518, CLIP normalization, nearest-neighbor mask resize, and binary masks. Their only intended method difference is the checkpoint/forward mechanism; C0 and C05 are generated from one CIR forward per cell.

## Known train/deploy difference

The training segmentation probability path (`tools/cir_rmt/runtime.py::_training_probability`) bilinearly resizes patch logits, averages stages, and applies softmax. The frozen deployment path (`model/phase2b_runtime.py::deploy_native_logits`) applies the domain deployment Gaussian blur first, then aligned bilinear resize, stage mean, and softmax. This difference is common to the parent and CIR deployment comparison and is not evidence that RMT itself caused a failure. It remains a possible train-vs-deploy contributor and must be tested numerically before changing an operator.

Medical image scoring is frozen and shared: `0.5 * classification_probability + 0.5 * pixel_max`. The exact evaluator consumes full-resolution disk-backed score/mask spools after worker shutdown and model teardown. The evaluator returns undefined image AUROC/AP for the Colon targets; those values are preserved as `None`, not converted to zero.

## Bounded numerical diagnostic

The corrected E14 VisA training-batch audit found mean absolute map difference 0.0032489155, maximum difference 0.9993287325, and Pearson correlation 0.6475440882. The training-side map scored Pixel AUROC/AP 0.9978585715/0.7237013607; the deployed map scored 0.9964954573/0.5963786302. This establishes a material path difference on the sampled batch, not its causal share of the Medical gap.

## Interpretation

No preprocessing or deployment change was made for the corrective run. The pipeline audit therefore supports the validity of the P/C0/C05 comparison within the current frozen evaluation protocol, while separating a known train/deploy operator mismatch from the parent-versus-CIR optimization comparison. The bounded operator check is complete; a full cross-target causal attribution remains unrun and must precede any operator edit.
