# Anchor loss scale audit

Status: `ILL_CONDITIONED`.

The current `ImageParameterAnchor` computes an equal-weighted mean over 99 image-adapter tensors of

```text
||theta_i - theta_ref_i||^2 / max(||theta_ref_i||^2, 1e-12).
```

The audit used the historical H2 E1 reference, shared E0, RA E10, and RA E16. The requested RA E1 and RA E4 checkpoints do not exist because RA saved candidate checkpoints only at E10/E12/E14/E16/E18/E20; those missing states were not reconstructed.

| Snapshot | Mean unweighted Anchor loss | Weighted by lambda=.001 | Zero/near-zero reference tensors |
|---|---:|---:|---:|
| Historical H2 E1 | 0.000000 | 0.000000 | 12 / 99 |
| Shared E0 | 1.248128 | 0.001248 | 12 / 99 |
| RA E10 | 6837.102618 | 6.837103 | 12 / 99 |
| RA E16 | 4099.349197 | 4.099349 | 12 / 99 |

The zero/near-zero reference set is:

```text
dfg_ss2d_branches.0.pre_norm.bias
dfg_ss2d_branches.0.ss2d.direction_logits
dfg_ss2d_branches.0.post_norm.bias
dfg_ss2d_branches.1.pre_norm.bias
dfg_ss2d_branches.1.ss2d.direction_logits
dfg_ss2d_branches.1.post_norm.bias
dfg_ss2d_branches.2.pre_norm.bias
dfg_ss2d_branches.2.ss2d.direction_logits
dfg_ss2d_branches.2.post_norm.bias
dfg_raw_gamma.0
dfg_raw_gamma.1
dfg_raw_gamma.2
```

At RA E10, the six zero-reference DFG pre/post-norm biases account for approximately 99.98% of the raw unweighted Anchor loss. The largest single terms are `dfg_ss2d_branches.0.post_norm.bias` (27.6646%), `.2.pre_norm.bias` (22.9253%), `.2.post_norm.bias` (14.4503%), `.1.pre_norm.bias` (13.5367%), `.0.pre_norm.bias` (12.8993%), and `.1.post_norm.bias` (8.5066%). RA E16 shows the same dominance pattern; `direction_logits` also becomes non-negligible relative to ordinary terms.

## Gradient evidence

The gradient probe used one fixed VisA training batch (`batch_size=6`, `shuffle=False`, `num_workers=0`, seed `12345`, native training segmentation, AMP enabled), with no optimizer step. It measured the raw Anchor gradient, the H2 base-task gradient, their cosine, and the required `lambda_anchor * ||g_anchor|| / ||g_task||`.

| Snapshot | `||g_anchor||` | `||g_task||` | `lambda*||g_anchor||/||g_task||` | Cosine |
|---|---:|---:|---:|---:|
| Historical H2 E1 | 0.000000 | 0.402939 | 0.000000 | n/a |
| Shared E0 | 0.271612 | 5.717946 | 0.0000475 | 0.001965 |
| RA E10 | 16619265.000000 | 0.414767 | 40068.918492 | approximately 0 |
| RA E16 | 12867933.000000 | 0.413246 | 31138.683816 | approximately 0 |

The RA E10 Anchor gradient is entirely concentrated in `dfg_ss2d_branches`; its base-task gradient on the fixed batch is zero in that family. The raw Anchor loss and gradient evidence agree: the per-tensor relative normalization with a `1e-12` clamp is not scale-safe for these zero-reference tensors. The problem is not inferred from loss magnitude alone.

The complete per-parameter and per-family tables are in `ANCHOR_PARAMETER_CONTRIBUTIONS.csv` and `ANCHOR_GRADIENT_DECOMPOSITION.csv`. No corrected Anchor formulation was implemented or trained in this audit. A future fix should use one globally normalized distance (or another explicitly justified zero-reference policy), with the exact formulation preregistered before training.
