# P5-FR1 method conformance review

The recovery retains the original P5-F method contract: industrial MVTec
only, 1725 canonical TEST identities, the frozen Phase2B deployment outputs,
the authoritative B1 same-image peer selector, four parameter-free geometry
families, 26 preregistered configurations, fixed folds, fixed matching,
fixed risk/triage budgets, fixed class bootstrap, and the original G1--G4
decision tree.

The only implementation clarifications are:

1. A peer Gram is stored as eight direct diagonal entries followed by the 28
   upper-triangle off-diagonal entries in `np.triu_indices(8, 1)` order.
2. All Gram packing, decoding, validation, and tests use the same shared
   constants and helpers.
3. Centering uses `H=I-11^T/K`, `b=H(c-Gw)`, and
   `t=1-2w^Tc+w^TGw`.
4. ASR uses outside centered-subspace energy divided by total query-centered
   energy; PGM uses the frozen sample-covariance whitened coordinate formula.
5. CSRC uses exact Kendall tau-b and Spearman degenerate rules.
6. The authoritative Phase5 percentile helper is reused and parity-tested.

These clarifications repair compact representation and numerical semantics;
they do not change the scientific hypothesis, candidate set, family search,
configuration count, selection, or gates. The post-GT evaluator is a neutral
consumer of frozen GT-free evidence and does not invoke the family modules.

## Conformance checklist

- [x] Previous P5-F invalid terminal remains immutable.
- [x] Dataset/checkpoint/config/canonical identity provenance is reused.
- [x] Prior partial attempt is not resumed; P5FR1 starts a fresh cache.
- [x] Exactly 26 configurations remain: PCRR 8, CSRC 8, ASR 6, PGM 4.
- [x] K=8, radius=3, percentile/risk/triage/matching/fold rules remain frozen.
- [x] GT-free runner has no mask/label performance path.
- [x] GT access is deferred until after the GT-free commit and backup.
- [x] No training, medical evaluation, E1, tuning, or alternative candidate.
