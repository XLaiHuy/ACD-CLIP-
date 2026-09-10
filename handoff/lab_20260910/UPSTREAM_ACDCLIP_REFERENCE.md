# Upstream ACD-CLIP Reference

Reference repository: `https://github.com/upupmake/ACD-CLIP`

Reference branch: `main`

Reference commit resolved on 2026-09-10:
`2685a6633d5466ff0255733d799ab9ce7b65a59a`

The public README at this reference documents the public ACD-CLIP workflow:

- default training horizon: 20 epochs;
- default batch size: 6;
- default input size: 518;
- Adam optimizer;
- image learning rate: 1e-3;
- text learning rate: 5e-4;
- StepLR decay factor: 0.9;
- one adapter checkpoint saved after each epoch;
- checkpoint iteration during testing.

The public repository also identifies the OpenAI CLIP ViT-L/14-336 download by
the SHA256 URL recorded in this handoff. The local base-weight SHA256 is
`3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`.

## Difference from this ACD-CLIP++ workspace

The current frozen implementation intentionally carries the ACD-CLIP++
research configuration. The handoff experiments retain, among other details:

- three adapter groups;
- current Dual-Branch DFG attention configuration with attention dimension
  256, temperature 8.0, SS2D, weight-residual fusion, and warmup beta;
- hybrid soft prompt settings and current regularizers;
- deterministic seed policy, AMP, gradient checkpointing, and the current
  operational worker setting;
- Family-Safe Anchor with the frozen shared E1 reference;
- CIR disabled and hard-background ranking disabled for both fresh runs;
- exact benchmark-equivalent Medical evaluation with identity/no TTA and
  pixel stride 1.

The upstream defaults are a provenance reference, not a replacement for the
frozen ACD-CLIP++ configuration. The public paper is not used here to claim
that best Medical epoch selection was preregistered. The two handoff runs are
explicitly target-selected exploratory transfer experiments.
