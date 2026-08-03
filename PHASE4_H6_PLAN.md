# Phase 4 H6 Progress 1 plan

## Scope lock

This branch implements only Progress 1: CoPS-style local state updates, a
global class VAE, image-conditioned dynamic soft contexts, four paired
normal/abnormal semantic factors, patch-wise Top-K text routing, and local H6
residual anomaly logits. It intentionally does not include visual experts,
visual routing, or cross-view consistency.

## Integration contract

- `ACDCLIP(image)` still returns `(seg_tokens, det_tokens)`.
- `ACDCLIP(image, return_phase4_features=True)` adds normalized segmentation
  tokens, pre-L2 segmentation tokens, detection tokens, and the final projected
  visual CLS token from the same visual forward.
- H6 creates one `e_factor[G,B,M,768,2]` bank. Its spatial mean routes create
  Phase2B global text; its patch routes create local residual logits.
- H6 never creates a parallel static-soft prediction bank. `ctx_normal` and
  `ctx_abnormal` are trainable base contexts for the dynamic update only.
- Hard prompts in `dataset/info.py` remain untouched. Adapted hard text is used
  for prediction; a detached non-adapted hard embedding is the Kg anchor.

## Locked Progress 1 configuration

- `n_groups=3`: image blocks `8,16,24`; text blocks `4,8,12`.
- Phase2B baseline: Conv-LoRA rank `16`, conv rank `8`, kernels `3,5`, image
  and text adapter weights `0.2`; DFG attention dimension `256`, tau `8`,
  SS2D `weight_residual`, beta warm-up `0,0.05,0.10`.
- Dynamic core: `M=4`, `K=2`, bank dimension `256`, router hidden `128`, VAE
  `768 -> 512 -> 256`, context length `4`.
- Hybrid alpha follows the verified Phase2B hybrid schedule: epochs `1-3=0`,
  epoch `4=.05`, epoch `5=.10`, epoch `6+=.20`.
- Progress 1 loss: task + `.10 center + .05 VAE reconstruction + betaKL VAE
  KL + .001 Kg + .001 orthogonal + .01 balance`; `lambda_k=0`.

## Numerical and training rules

- Train from OpenAI CLIP only; Phase2B and any other adapters are not loaded.
- OpenAI CLIP base parameters remain frozen, but the visual forward remains in
  autograd so Phase2B Conv-LoRA/adapters receive gradients.
- BF16 is the default outer autocast. State/VAE/router reductions and the H6
  cosine/logit path use FP32. BF16 is rejected on unsupported CUDA hardware.
- Batch size `1`, accumulation `6`, 20 epochs, image size `518`, workers `6`,
  pin memory disabled by default. No silent image-size or metric changes.

## Deferred interfaces

Progress 2 can add visual experts behind the existing H6 router contract.
Progress 3 can add two-view data and consistency losses without changing the
Progress 1 checkpoint fields or Phase2B visual return contract.
