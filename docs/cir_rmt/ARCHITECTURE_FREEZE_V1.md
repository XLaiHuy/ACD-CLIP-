# CIR_DFG_RMT_V1 architecture freeze

Status: **FROZEN**

This document is the architecture boundary for the CIR worktree. The
architecture search is closed. CIR is one unified, exact score-space
mechanism on top of the native DFG path; it is not a second training stage.

## Identity

- `arch_id`: `CIR_DFG_RMT_V1`
- `architecture_version`: `1`
- parent: canonical DFG adapter with `n_groups=3`
- peer count: `K=8`
- transport: KL-antisymmetric, with one global `rmt_transport_alpha`
- score mode: `exact_score_space`
- gradient rule: peer search and `delta` are detached; native DFG and final
  score remain differentiable

## Frozen mechanism

```text
native DFG (GAP + SS2D weights)
                 |
                 v
       8 GT-free normal-like peers
                 |
                 v
          peer anomaly margins
                 |
                 v
          midpoint Median + MAD
                 |
                 v
       observed-vs-peer signed deviation
                 |
                 v
                tanh
                 |
                 v
             delta in (-1,1)
                 |
                 v
      KL antisymmetric transport
          normal down / abnormal up
                 |
                 v
            exact score-space
                 |
                 v
             anomaly map
```

For observed patch feature `x_p`, native text prototypes are `T_N` and `T_A`.
The observed anomaly margin is

```text
m_obs = cos(x_p, T_A) - cos(x_p, T_N)
```

The peer margins are `m_peer[k]`. For even `K=8`, the center is the midpoint
median, not the lower-median behavior of `torch.median`:

```text
center = (x_(4) + x_(5)) / 2
MAD    = midpoint_median(abs(m_peer - center))
d      = 1.4826 * MAD
z_{g,p}      = (m_obs_{g,p} - center_{g,p}) / (d_{g,p} + rmt_eps)
delta_{g,p}  = tanh(z_{g,p})
```

The robust statistic is computed independently for each native
stage/group depth. For an output stage s, the transport vector is
delta_{g,p} across the three groups, so a scalar patch offset is never
added to every softmax entry:

```text
wA[s,p,g] = softmax_g(log(wA_native[s,p,g]) + rmt_transport_alpha * delta[g,p])
wN[s,p,g] = softmax_g(log(wN_native[s,p,g]) - rmt_transport_alpha * delta[g,p])
```

delta is stop-gradient. Peer search uses pooled native margins only to define
the normal-like candidate pool; the signed per-stage margins above remain the
evidence used for transport. No Trust, Need, FU, router, expert, decoder,
second CLIP, selector, target-specific repair, or auxiliary CIR head is part
of this architecture.

## Exact score-space contract

The reference path constructs the weighted text prototype first:

```text
t_pc = normalize(sum_g w_pgc * t_gc)
score_pc = 10 * dot(x_p, t_pc)
```

The optimized path avoids materializing large descriptor tensors. It uses:

```text
C_c[g,h] = dot(t_gc, t_hc)
a_pgc    = dot(x_p, t_gc)
numerator   = sum_g w_pgc * a_pgc
denominator = sqrt(w_pc^T C_c w_pc + eps)
score_pc    = 10 * numerator / denominator
```

The two paths must match in FP32 within the documented tolerance. A weighted
sum of independently normalized group scores is not an allowed substitute.

## Data and release boundary

Peer search is image-only and GT-free. Peer indices are non-differentiable;
they are built from detached features, with self exclusion and deterministic
ordering. Source selection may use VisA or MVTec according to the full-run
contract, but medical targets are never used to select `rmt_transport_alpha`.

Every CIR checkpoint and result records the architecture/config/git identity,
source dataset, epoch, group count, peer count, transport alpha, score mode,
checkpoint hash, and evaluator protocol. A mismatch is a hard failure.

The parent SABRA worktree is reference-only. All CIR implementation and
artifacts live in the `cir_rmt` namespace and under
`runs/cir_rmt/CIR_DFG_RMT_V1/`.
