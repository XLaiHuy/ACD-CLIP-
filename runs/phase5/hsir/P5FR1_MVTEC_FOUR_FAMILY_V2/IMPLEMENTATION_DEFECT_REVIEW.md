# P5-FR1 implementation defect review

## Scope

This review compares the prior P5-F implementation at `859d4e2` with the
frozen P5-F protocol and the P5-FR1 geometry contract. It is written before
any P5-FR1 official MVTec forward and before any GT access. The prior P5-F
worktree, cache, and result remain immutable.

## Findings

| Area | Prior implementation | Classification | Required disposition |
|---|---|---|---|
| Gram packing | The writer persisted only the 28 off-diagonal values while the validator/decoder contract required eight diagonal values followed by 28 off-diagonal values. | `IMPLEMENTATION_FIX` | Add shared `pack_gram`/`decode_gram` with the frozen `diag8_then_offdiag28` layout and validate the direct Gram diagonal. |
| Gram construction | The writer used pairwise products only and did not persist the direct full peer Gram. | `IMPLEMENTATION_FIX` | Compute `P @ P.T` directly, then pack it through the shared helper. |
| Centered query geometry | The family code used `c - mean(c)` and `1 - 2 mean(c) + mean(G)`. | `IMPLEMENTATION_FIX` | Use `b = H(c-Gw)` and `t = 1-2w^Tc+w^TGw`. |
| ASR | Projection used the incorrect cross-vector and the old energy quantity. | `IMPLEMENTATION_FIX` | Apply the frozen centered-subspace residual equations using `b` and `t`. |
| PGM | Whitening used the incorrect cross-vector and a ridge-like denominator. | `IMPLEMENTATION_FIX` | Apply `(K-1)(v^Tb)^2/lambda^2`, with no shrinkage or sweep. |
| CSRC Kendall | Both-tied pairs were counted in both tie totals, and constant vectors returned zero unconditionally. | `IMPLEMENTATION_FIX` | Implement exact tau-b pair accounting and frozen constant-vector rules. |
| CSRC Spearman | Constant vectors returned zero even when both vectors were identical constants. | `IMPLEMENTATION_FIX` | Apply both-constant=`1.0`, one-constant=`0.0`, otherwise average-tie Pearson correlation. |
| Percentile ranks | A second local implementation was used instead of the authoritative helper. | `IMPLEMENTATION_FIX` | Reuse `audit_phase5_hsir.percentile_rank` and parity-test it. |
| PCRR | The transform formula and search count are unchanged, but it relied on the malformed compact Gram and local pair-index construction. | `IMPLEMENTATION_FIX` | Consume the corrected shared decoder/constants only. |
| Invalid references | The zero-evidence/separate-validity behavior is correct and remains frozen. | `IMPLEMENTATION_FIX` | Preserve exactly while validating all four families. |
| Dataset, checkpoint, config, B1 selector, folds, budgets, gates | No intended scientific change. | `NO_CHANGE` | Reuse prior authoritative provenance and protocol. |

## Scope decision

Every required change above is a correction to the implementation of the
already-frozen geometry contract. No `SCIENTIFIC_CHANGE` was identified. The
P5-FR1 run is blocked if later inspection reveals a change to a frozen family,
selection, matching, budget, bootstrap, or gate definition.

## Barriers

- No MVTec model forward is performed by this review.
- No mask, label, or ground-truth field is read.
- The old `/tmp/p5f_mvtec_common` and `/tmp/p5f_mvtec_all_config_evidence`
  namespaces are not read or repaired.
- The P5-FR1 cache namespace is required to start empty and independently
  provenance-bound.
