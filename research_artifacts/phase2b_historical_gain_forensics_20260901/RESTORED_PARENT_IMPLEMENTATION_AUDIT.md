# Restored-parent implementation audit

Status: NOT_RUN for a new active-branch restored-H2 trainer. The exact historical source and formula are recovered, but the active branch has not yet introduced a restored-H2 training variant. This prevents an unsupported claim that current C2 code is H2-equivalent.

The implementation gate must preserve historical AMP/mixed precision, hybrid alpha and prompt policy, KG/K-reg coefficients, Adam group order/rates, StepLR timing, DFG fields, data path/augmentation, and historical loss/scoring semantics. Engineering improvements may add atomic writes, resume state, and bounded ledgers only after fixed-input parity.
