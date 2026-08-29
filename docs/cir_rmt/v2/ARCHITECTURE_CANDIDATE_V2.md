# CIR_DFG_RMT_V2 architecture candidate

Status: **CANDIDATE — NOT RELEASE FROZEN**

`CIR_DFG_RMT_V2` is identical to `CIR_DFG_RMT_V1` in the parent canonical
Phase2B/DFG setup, `G=3`, `K=8` GT-free shared peer construction,
per-stage/per-group observed margins and detached robust delta, exact
score-space scoring, losses, optimizer, evaluator, and source/target protocol.

The one and only scientific change is the antisymmetric transport direction.
V1 used:

```text
wA* = softmax_g(log(wA_native) + alpha * delta)
wN* = softmax_g(log(wN_native) - alpha * delta)
```

V2 candidate uses a non-negative transport magnitude and the explicit
`rmt_transport_direction = "abnormal_minus_normal_plus"` contract:

```text
wA* = softmax_g(log(wA_native) - alpha * delta)
wN* = softmax_g(log(wN_native) + alpha * delta)
```

The tensor contract remains `m_obs [S,B,P,G]`, shared peer indices
`[B,P,K]`, peer margins `[S,B,P,K,G]`, and detached `delta [S,B,P,G]`.
No new branch, head, loss, selector, router, expert, decoder, second CLIP,
target calibration, or architecture component is introduced.

The V2 direction remains a candidate until the preregistered 120-image
VisA-only source confirmation is complete. Alpha remains provisional.
