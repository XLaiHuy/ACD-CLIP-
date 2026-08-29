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

For each visual output stage `s`, batch item `b`, patch `p`, and text group
`g`, the feature and native prototypes are `x[s,b,p]` and `T_N[b,g]`,
`T_A[b,g]`. Visual-stage and text-group axes are distinct.
The unfused observed anomaly margin is

```text
m_obs[s,b,p,g] = cos(x[s,b,p], T_A[b,g]) - cos(x[s,b,p], T_N[b,g])
```

Peer selection produces one shared, GT-free index set `i[b,p,k]` with shape
`[B,P,K]`, `K=8`, from detached pooled features and group-pooled margins. The
gathered peer margins retain both independent axes:

```text
m_peer[s,b,p,k,g] = m_obs[s,b,i[b,p,k],g]       # [S,B,P,K,G]
center[s,b,p,g] = midpoint_median_k(m_peer[s,b,p,:,g])
MAD[s,b,p,g] = midpoint_median_k(abs(m_peer[s,b,p,:,g] - center[s,b,p,g]))
d[s,b,p,g] = 1.4826 * MAD[s,b,p,g]
z[s,b,p,g] = (m_obs[s,b,p,g] - center[s,b,p,g]) / (d[s,b,p,g] + rmt_eps)
delta[s,b,p,g] = tanh(z[s,b,p,g])                    # [S,B,P,G]
```

The robust statistic is reduced over K independently for every
visual-stage/patch/group coordinate `[s,b,p,g]`. Transport applies a group
softmax separately at each stage and patch; native weights `[s,b,g]` broadcast
over patches, while the evidence remains `[s,b,p,g]`:

```text
wA*[s,b,p,g] = softmax_g(log(wA_native[s,b,g]) + rmt_transport_alpha * delta[s,b,p,g])
wN*[s,b,p,g] = softmax_g(log(wN_native[s,b,g]) - rmt_transport_alpha * delta[s,b,p,g])
```

delta is stop-gradient. Peer search uses pooled group margins only to define
the normal-like candidate pool; the unfused per-stage/per-group margins above
remain the evidence used for transport. The stage axis `s` and text-group axis
`g` are never mapped or broadcast into one another, and no fused native DFG
margin is used as RMT evidence. No Trust, Need, FU, router, expert, decoder,
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
