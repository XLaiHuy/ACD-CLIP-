# P5-D0 graph non-conformity audit design review

This audit is a cache-only diagnostic of all certified R0 same-image relations. It does not change the Phase2B predictor, the R0 selector, P5B action, deployment, training, or any candidate rule.

## Adversarial review and frozen resolutions

- **GT leakage:** graph edges, node signals, hashes, and all Hodge quantities are materialized before masks or labels are read. GT is joined only in a second post-hoc pass.
- **Selected-versus-certified confusion:** the primary graph uses every R0 `certified` relation, before disjoint greedy selection. The archived selected trace is checked separately for parity and is not substituted for the graph.
- **Orientation ambiguity:** for every strict base inversion `(i,j)` with `m_i < m_j`, the edge is oriented `i -> j`, with frozen target `g=q_m(j)-q_m(i)>0` and fitted equation `p_i-p_j ~= g`.
- **Rank/tie ambiguity:** the base percentile uses the audited float32 descending-score, ascending-patch-ID order. Ties are resolved by patch ID; no new threshold or tie policy is introduced.
- **Raw-score confounding:** raw `m_j-m_i` is retained only as a diagnostic. The primary graph target is the exact within-image base percentile gap `q_m(j)-q_m(i)`.
- **Hodge gauge and numerical instability:** each edge-connected component is solved in float64 with a zero-mean gauge. Projection identity, residual orthogonality, and deterministic edge-order tests are mandatory.
- **Disconnected graphs:** components are independent; isolated nodes receive zero potential and zero incident-edge diagnostics. S6 is the deterministic percentile/rank of the full-image potential vector, not a selected post-hoc transform.
- **Signal selection bias:** S6 is preregistered as primary. S1-S5 and S7 are diagnostic-only and cannot replace S6 or determine a different primary signal.
- **Shift asymmetry:** aligned and shifted use the exact frozen shift of evidence only. Base scores, D_rank, validity, cells, and graph target semantics remain unchanged. Shifted results are a control, not a second opportunity to select a signal.
- **Deployment overreach:** graph leverage is not treated as deployment benefit. Any future design must separately respect native correction followed by blur, bilinear resize, stage mean, and softmax.
- **Threshold fishing:** no rank, spatial, score, AUC, or AP cutoff is searched. The only decision gates are the frozen G0-G4 rules in `PROTOCOL.json`.
- **Redundancy:** S6 is compared with frozen base score, D_rank, and E_nonlocal using preregistered rank correlations. A high correlation can reject novelty; it cannot be used to tune S6.
- **Class imbalance:** class is the bootstrap unit, with 2,000 repetitions and seed 7701 inherited from the frozen forensic protocol.
- **Runtime/provenance:** only the finalized `/tmp/p5_r0_run2` cache is read; model forwards and training are zero. Protected source hashes and cache manifest SHA are recorded.

## Decision boundary

The audit can support only the existence of graph non-conformity leverage for a later design question. It cannot select or implement D1, change R0/P5B, or claim causal deployment improvement.
- **Protocol/code mismatch found in prior result:** the previous implementation used `m_bar_j-m_bar_i` as Hodge flow and persisted incident S1-S4 definitions, although the frozen contract requires `q_m(j)-q_m(i)` and target-only S1-S4. The prior graph cache/result is therefore invalidated; protocol v2 freezes the corrected flow, target-only signals, and explicit S7 epsilon/zero-degree convention before rerunning.
- **Sparse-solver availability:** the preferred sparse dependency is unavailable in the execution environment. The corrected audit uses the existing deterministic float64 reduced-Laplacian NumPy solve with a documented fallback only for singular numerical failure; no model or protected-source dependency is added.
- **Post-hoc completeness:** Pearson and Spearman nonredundancy, normal-image S6 distributions, and diagnostic-signal correlations with S6/base are persisted explicitly. These remain descriptive and cannot alter the primary signal or gates.
