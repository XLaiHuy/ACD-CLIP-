# Phase4 Progress 1 v2 Audit

## Repository State

- Base branch inspected: `phase4-progress1-cops-dynamic-prompt`
- V2 branch: `phase4-progress1-v2-specialization-fix`
- Local base commit before v2 work: `0623912 fix(phase4): make exact medical test memory safe`
- Remote: `origin https://github.com/XLaiHuy/ACD-CLIP-.git`
- Training executed by Codex for v2: NO
- Medical test executed by Codex for v2: NO

Known non-code workspace noise left unstaged:

- `model/ViT-L-14-336px.pt`: hydrated OpenAI/LFS weight in working tree
- `runs/phase4/`: local run outputs

## Existing P1-v1 Run Facts

Found run path:

- `runs/phase4/progress1_cops_dynamic_prompt_seed0_retry1`

Artifacts found:

- `train.log`
- `adapter_1.pth` through `adapter_20.pth`
- `medical_validation_selection.json`
- `medical_test_results_by_dataset.csv`

Existing P1-v1 test macro from `medical_test_results_by_dataset.csv`:

- Pixel 6 macro AUROC/AP: `81.6776 / 22.7672`
- Image 3 macro AUROC/AP over Brain/Liver/Retina: `73.4533 / 75.3233`

Interpretation boundary:

- These are facts from existing result files.
- They do not prove which implementation component caused the drop by themselves.
- Tqdm postfix values are treated as last-batch diagnostics, not epoch means.

## Critical Audit Findings

### 1. Pre-fusion L2 Norms

Status: SUPPORTED.

Source evidence:

- `model/h6/model.py` previously fused `hard_adapted.unsqueeze(2)` with `dynamic` and normalized only after mixing.
- Hard prompt helpers normalize averaged prompt features, but dynamic text comes directly from `encode_dynamic_prompt_text` level outputs and was not explicitly normalized before fusion.

Fix:

- `hard_adapted`, `hard_frozen`, and `dynamic_text` are now explicitly L2-normalized before fusion, KG anchor loss, and residual-diversity loss.
- `H6Progress1._fuse_factor_bank()` defines the alpha semantics and is unit-tested for alpha endpoints.

### 2. Frozen Hard Anchor Independence

Status: SUPPORTED.

Source evidence:

- `utils.get_hard_anchor_single_class_text_embedding()` previously called `model.encode_text(..., adapt_text=False)`.
- `model/adapter.py::_encode_text_from_embeddings()` still applied `self.text_adapter["layer_norms"]` even when `adapt_text=False`.
- Therefore the old hard anchor bypassed text LoRA/AddWeight but still depended on trainable text adapter LayerNorms.

Fix:

- `ACDCLIP.encode_frozen_anchor_text()` now bypasses text LoRA, text adapter weights, and trainable text adapter LayerNorms.
- It uses fixed functional layer normalization under `torch.no_grad()`, then normalizes outputs.
- Hard anchors are cached by `(dataset, class, device)` after this frozen path.

### 3. Eta Class Gate

Status: ACCEPTED MODIFICATION.

Decision:

- No `eta_class` interpolation gate was added.

Fix:

- `ClassVAE` still samples `z` for reconstruction training.
- The prompt semantic path uses deterministic `decoder(mu)`.
- `gamma_class` and `hybrid_alpha` remain the only class-prompt contribution gates.

### 4. Residual Diversity Target

Status: SUPPORTED.

Source evidence:

- The previous `factor_orthogonal_loss()` operated on final hard-dynamic mixed `factor_bank`.
- With `hybrid_alpha=0.2`, final factors contain a large common hard semantic component.

Fix:

- Added `dynamic_residual_diversity_loss(dynamic_text, hard_frozen)`.
- Training now uses `h6_batch["residual_diversity"]` for the `lambda_h6_orth` slot.

### 5. Concept-Key Router and Dense/Sparse Views

Status: SUPPORTED.

Fix:

- Router logits are now cosine dot products between projected patch queries and
  CoPS concept keys.
- The router always returns both dense and sparse probability views.
- Prediction uses dense probabilities through epoch 6 in the v2 script, then
  sparse Top-K probabilities from epoch 7.
- Diagnostics report prediction, dense, sparse, and selected Top-K factor usage.

### 6. Factor-Aware Center Loss and KL Schedule

Status: SUPPORTED.

Fix:

- P1-v2 training uses factor-aware center loss weighted by detached dense
  routing probabilities.
- Routing balance also uses dense probabilities.
- VAE KL is zero for the first four epochs in the v2 script, then linearly
  warms to `1e-4`.

## Remaining Uncertainty

- Router collapse, factor assignment quality, and VAE posterior-collapse risk require v2 training logs to confirm or reject.
- KL near zero alone is not treated as confirmed posterior collapse.
- Full metric impact is not claimed until real v2 training and no-leakage
  validation-selected medical testing are complete.
