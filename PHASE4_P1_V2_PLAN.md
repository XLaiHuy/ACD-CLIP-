# Phase4 Progress 1 v2 Plan

This is Progress 1 v2, not Progress 2. It keeps the Phase2B + H6 macro-architecture and applies stability fixes to text fusion, frozen anchors, VAE prompt semantics, and residual diversity.

| Proposed fix | Evidence from current code/log | Expected benefit | Main risk | Exact files/functions affected | Decision |
|---|---|---|---|---|---|
| Normalize hard/dynamic text before hybrid fusion | Dynamic text was encoded directly from dynamic prompt contexts and final mix was normalized only after interpolation | Make `hybrid_alpha` interpretable and stabilize KG/residual losses | May slightly change old P1-v1 numeric behavior | `model/h6/model.py::H6Progress1._encode_dynamic_bank`, `_fuse_factor_bank`, `build_batch` | accept |
| Explicit frozen anchor encoding mode | `adapt_text=False` still used trainable text adapter LayerNorms | Anchor remains independent of trainable text parameters | Functional LN scale may differ from old adapter LN | `model/adapter.py::encode_frozen_anchor_text`, `utils.get_hard_anchor_single_class_text_embedding` | accept |
| Cache frozen hard anchors | Frozen path is now independent of trainable adapters | Avoid repeated hard-prompt text encoding in H6 batches | Cache key must include device | `utils.get_hard_anchor_single_class_text_embedding` | accept |
| Add `eta_class` interpolation | User critical audit warns this would over-suppress VAE path | None for first v2 | Third gate can hide class update | not implemented | reject |
| Use sampled z for VAE reconstruction only, decoder(mu) for prompt | Current `ClassVAE` used sampled `z` as `class_semantic` in train mode | Deterministic prompt semantic with stochastic auxiliary VAE | Reconstruction and prompt paths diverge; must log both if diagnosing | `model/h6/semantic_bank.py::ClassVAE.forward` | accept |
| Diversity on dynamic residual directions | Final factor bank is mostly hard semantic when alpha is small | Encourage factor-specific dynamic corrections instead of orthogonalizing common hard prompt | Zero residual can make diversity weak early | `model/h6/losses.py::dynamic_residual_diversity_loss`, `train.py::train_h6_progress1` | accept |
| Full factor-aware center loss | Existing center loss picks nearest prototype over all factors | Stronger router/prototype correspondence | Bigger behavior change; monitor loss scale | `model/h6/losses.py::factor_aware_center_loss`, `train.py::train_h6_progress1` | accept |
| Concept-key router redesign | Current router used independent MLP logits; concept keys were not routing keys | Better semantic correspondence | Wider router contract change; checkpoint config must match | `model/h6/router.py::PatchRouter`, `model/h6/model.py::H6Progress1.build_batch` | accept |
| Longer dense routing warm-up | P1-v1 could activate Top-K before dynamic factors specialize | Reduce early dead factors | Delays sparse specialization | `scripts/phase4/train_progress1_v2.sh` uses `--h6_router_soft_epochs 6` | accept |
| Exact medical test memory safety | Brain exact test was OS-killed after full progress due RAM accumulation | Preserve exact metric while avoiding RAM kill | Disk-backed merge is slower | `test.py`, `scripts/phase4/test_6medical_exact.sh` | already implemented |

## V2 Behavior Summary

- Text fusion inputs are L2-normalized before mixing.
- Hard anchor path bypasses all trainable text adapter modules.
- VAE prompt semantic is `decoder(mu)`.
- Sampled VAE latent is used only for reconstruction loss.
- No `eta_class` gate is added.
- Residual diversity is computed against frozen hard anchors.
- Router logits are concept-key dot products.
- Center loss is factor-aware and uses detached dense router probabilities.
- Routing balance uses dense probabilities, while prediction switches to sparse Top-K after dense warm-up.
- VAE KL is zero for the first four epochs, then linearly warms to `1e-4`.
- V2 training script keeps `M=4`, `K=2`, and delays Top-K until after six dense-routing epochs.

## Acceptance Criteria After Real Training

Do not claim success until real v2 logs/results exist. Review:

- dense and sparse factor usage
- dead factor counts
- residual diversity not locked to a constant
- dynamic residual norms finite and non-identical
- dynamic-hard cosine controlled
- VAE `mu`, `logvar`, `decoded_mu` statistics
- exact medical metrics under the same protocol as P1-v1 and Phase2B references
