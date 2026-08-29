# CIR_DFG_RMT_V2 architecture freeze

Status: **FROZEN** (transport direction only)

The bounded, preregistered VisA-only source confirmation passed. This document
freezes the V2 architecture direction; `rmt_transport_alpha` remains
**PROVISIONAL** and no release lock is implied.

## Identity and scope

- `arch_id`: `CIR_DFG_RMT_V2`
- `architecture_version`: `2`
- parent: canonical Phase2B/DFG adapter with `n_groups=3`
- peer count: `K=8`
- score mode: `exact_score_space`
- gradient rule: peer search and `delta` are detached; native DFG and final
  score remain differentiable
- only scientific change from V1: antisymmetric transport direction

All tensor geometry, peer construction, robust statistics, scoring, losses,
optimizer, evaluator, and source/target protocol are unchanged from V1.
In particular, observed margins are `[S,B,P,G]`, shared peer indices are
`[B,P,K]`, peer margins are `[S,B,P,K,G]`, and detached evidence is
`delta [S,B,P,G]`.

## Frozen direction

V1 (the falsified frozen experiment) used:

```text
wA* = softmax_g(log(wA_native) + alpha * delta)
wN* = softmax_g(log(wN_native) - alpha * delta)
```

V2 uses the explicit non-negative magnitude
`rmt_transport_direction = "abnormal_minus_normal_plus"`:

```text
wA* = softmax_g(log(wA_native) - alpha * delta)
wN* = softmax_g(log(wN_native) + alpha * delta)
```

No negative-alpha encoding is permitted. No additional module, head, loss,
selector, router, expert, decoder, second CLIP, target calibration, or image
score change is part of V2.

## Source-only confirmation evidence

The confirmation used VisA only, seed `0`, a deterministic class-stratified
subset of 120 images (12 classes, 5 normal and 5 anomalous per class), and
the preregistered V2 grid `0/0.10/0.25/0.50`. The evidence is archived under
`runs/cir_rmt/CIR_DFG_RMT_V2/source_confirmation/`.

| alpha | pixel AUROC | pixel AP | image AUROC | image AP |
|---:|---:|---:|---:|---:|
| 0.00 | 0.430918 | 0.004577 | 0.503333 | 0.593155 |
| 0.10 | 0.454278 | 0.005489 | 0.506667 | 0.598975 |
| 0.25 | 0.486811 | 0.012185 | 0.496667 | 0.606554 |
| 0.50 | 0.523760 | 0.012816 | 0.530000 | 0.617897 |

The preregistered rule is satisfied: all three nonzero values improve both
pixel metrics, and alpha `0.10` (also `0.25`) keeps both image-level drops at
or below `0.02` versus alpha zero. This supports the direction on the bounded
source confirmation only; it does not claim target or full-training
superiority.

`RELEASE_LOCK = FALSE`. G2/G3 release gates, GPU G4, G5 smoke, and full
training are not authorized by this confirmation.
