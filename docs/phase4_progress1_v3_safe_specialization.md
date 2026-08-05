# Phase4 Progress1 v3 safe specialization

This note documents the conservative P1-v3 change set.  It preserves the P1
macro-architecture: OpenAI CLIP initialization only, one shared dynamic semantic
factor bank, M=4 paired normal/abnormal factors, concept-key dot-product router,
Top-K=2 final routing, frozen hard anchor, normalized text fusion, dynamic
residual diversity, and deterministic `decoder(mu)` prompt semantics.

## Observed v2 failure

The stopped v2 run showed dense routing with almost perfect global balance and
near-max entropy, while hypothetical or actual Top-K selected only one factor
pair for almost every patch.  This means low balance loss did not imply patch
specialization: dense probabilities were close to uniform, then the hard Top-K
argmax collapsed to a repeated pair.

The VAE also showed shrinking `mu_std` and KL.  KL alone is not enough to
diagnose posterior collapse, so v3 logs raw KL, effective KL, `mu_std`,
`decoded_mu_std`, and class semantic variance.

## Router changes

- Dense probabilities remain the semantic router distribution:
  `softmax(concept-key dot-product logits)`.
- Sparse probabilities are normalized Top-K probabilities with exactly K=2
  non-zero factors.
- Prediction uses deterministic interpolation:
  dense for epochs 1-8, then ratios 0.25/0.50/0.75 for epochs 9/10/11, and
  straight-through sparse from epoch 12 onward.
- Straight-through sparse gives sparse forward values while preserving dense
  router gradients.
- A detached state-aware prototype teacher trains dense router probabilities
  toward normal or abnormal prototypes selected by the training mask.
- Factor-aware center loss keeps the opposite direction: detached dense router
  assignment updates the corresponding prototypes/factors.
- A small EMA load bias affects only Top-K selection opportunity; within-Top-K
  weights still use semantic logits.
- Dense balance is kept as a weak batch/dataset anti-starvation term and is
  reduced to `0.001` in the v3 script.

## Failure detector

Readiness metrics do not delay sparse routing.  After `sparse_ratio >= 0.50`,
two consecutive completed epochs with either two or more sparse-dead factors or
one or fewer unique Top-K pairs save a diagnostic checkpoint and abort with
`h6_router_specialization_failed`.

## VAE protection

The KL schedule is delayed: epochs 1-8 use beta 0, epochs 9-12 ramp to
`1e-5`, then remain at `1e-5`.  Optimization uses scalar free bits in the same
units as the logged KL: sum over latent dimensions, mean over batch.

The class prompt semantic is a fixed safety blend:

```text
normalize((1 - ratio) * normalize(detached_cls24)
          + ratio * normalize(decoder(mu)))
```

The default ratio is `0.25`.  This is not a learnable third gate.  Sampled `z`
is still used only for reconstruction.

## Checkpoint contract

P1-v3 checkpoints use checkpoint version 3 and store explicit behavior fields:
`progress_version=P1-v3`, dense/sparse transition settings, router teacher
settings, load-bias settings and buffers, failure detector settings, center
detach, dynamic residual diversity, frozen anchor mode, KL schedule/free-bits,
and VAE class skip ratio.

## Protocol

The chained v3 script keeps the no-leakage protocol:

1. train on VisA;
2. sweep medical validation epochs using one common rule;
3. select one common best epoch from validation;
4. test medical exactly once at that selected epoch.

No medical label, validation, or test information is used during training.
