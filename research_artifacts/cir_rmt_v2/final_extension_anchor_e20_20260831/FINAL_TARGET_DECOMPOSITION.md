# Final target decomposition

The target decomposition keeps the three causal layers separate:

- Anchor training effect: A0 - P.
- Conditional inference RMT effect: A05 - A0.
- Total anchored CIR effect: A05 - P.
- Old-CIR comparison: A0 - C_OLD_0; this is a trajectory/protocol comparison, not a pure architecture effect.

Medical rows are exact evaluator outputs. P and C_OLD are reused frozen results; A0/A05 are the 72 new logical cells recorded by TARGET_EVAL_LEDGER.csv.

The decision recorded for this matrix is `KEEP_ANCHOR_DISABLE_INFERENCE_RMT_CANDIDATE`. No target epoch was selected after seeing target metrics; the primary source-only rule was frozen as A05 E20 before evaluation.

Rows in domain delta table: 36.
