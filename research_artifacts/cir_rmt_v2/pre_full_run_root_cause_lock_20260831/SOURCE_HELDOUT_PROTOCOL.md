# Source-only category-held-out protocol

Status: preregistered before the new diagnostic forward pass.

This is a bounded diagnostic of the already completed P/C0 checkpoints. It
does not train a new model, access Medical or MVTec data, tune a target, or
change the production evaluator.

## Fixed protocol

- Source: `/home/ai4/caohuy/data/VisA_20220922`, manifest
  `dataset/hub/VisA.jsonl`.
- Deterministic sample seed: `9014`.
- Sampling: eight images per category, balanced as four normal and four
  anomalous images where available; category-local deterministic shuffling is
  seeded by `9014 + sum(ord(character) for character in category)` and the
  selected rows are finally ordered by image path.
- Categories held out from the seen partition: `cashew`, `macaroni2`, `pcb3`,
  and `pipe_fryum`.
- Seen partition: the remaining eight VisA categories.
- Checkpoints: P and C0 at E10, E12, E14, E16, E18, and E20. C05 is used only
  for the already-defined inference compensation comparison.
- Metrics: exact repository tie-aware binary AUROC/AP for flattened pixel
  scores and image scores. Image scores use the frozen Industrial `.9/.1`
  classification/pixel-max fusion.
- No image from a held-out category is used for choosing a solution or its
  hyperparameter. No category labels or masks enter the GT-free RMT forward;
  masks are used only after inference for diagnostic metrics.

## Interpretation rule

The original P and C0 models were trained on all VisA categories. Therefore
this partition is an assessment split with category separation, not a true
unseen-training-category counterfactual. It can show a category-dependent
source pattern and can falsify a broad claim of uniform source behavior, but
it cannot alone prove training-category overspecialization.

`OVERSPECIALIZATION_SUPPORTED` requires all of the following in the compact
results: a consistent seen-versus-held-out direction across epochs/categories,
corresponding parameter or representation/module evidence, and no explanation
by an evaluator/deployment artifact. Otherwise the result is
`OVERSPECIALIZATION_UNPROVEN` or `NOT_SUPPORTED`.
