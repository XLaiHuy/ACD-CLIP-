# P25R Final Decision

Decision: `P25R_ENGINEERING_STOP`.

No P25R scientific attempt was created.  The required 128-candidate candle
direct-versus-fast parity cannot meet the frozen `<=1e-12` tolerance using the
permitted sparse-basis engine.  The first deterministic mismatch is panel row
15 (image 1, patch 907, signed action -1): direct pAP is
`0.501883222367168` and fast pAP is `0.5018832223692163`, an absolute error of
`2.048361480433414e-12`.

This is not an irrelevant reduction-order discrepancy.  The fast and direct
candidate support vectors have different stable score ordering, and their
float32 score-group inventories differ (13,075 versus 13,032 groups).  The
P25R clause permitting tiny numerical differences therefore does not apply.

The cause is float32 operation order: frozen deployment adds the correction to
native logits before Gaussian blur and bilinear resize, whereas the sparse
basis adds the independently deployed basis after the native deployment.  The
operator is mathematically linear but not bit-identical under float32 rounding.
Replacing the fast path with one full three-stage deployment per candidate
would violate the P25R execution contract.  No target V, Q1/Q2 result, model,
or policy result is interpretable.

Firewall: MVTec 0; Medical 0; additional CLIP forwards 0; Phase2B training
steps 0.
