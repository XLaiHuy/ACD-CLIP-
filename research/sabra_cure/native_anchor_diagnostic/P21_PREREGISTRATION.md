# P21 Native-Anchored Action-Space & Value Identifiability Diagnostic V1

Parent: `fcbd218c28cdd5c58f54e5d187f8d8ff9bdaa63f` (`P14_SCIENCE_RECOVERED_STOP`). P20 partial/history artifacts are immutable and P20's 12 science folds and 12 independent audits are reused cache-only.

P21 is one final post-hoc diagnostic, not a deployable method. It uses exactly one attempt, begins only from a published clean execution base, and stops after its terminal evidence is published.

Frozen constants: 12 VisA LOCO held-class order; alpha `0.25`; float32 score/tie semantics; AP strict improvement `>1e-12`; source-only `numpy.quantile(..., method="linear")`; no MVTec, Medical, CLIP forward, Phase2B step, prompt/adapter/encoder training, alpha sweep, R2-v3/R3/R4, target-domain memory, or test-time adaptation.

Route: Stage A reproduces P20. Stage B tests A0={NATIVE,SAFE20,EXPAND40}. Only if A0 lacks strong headroom, Stage C tests the sole additional action SAFE30. Only if A0 or A1 has strong headroom, Stage D runs P0/P1/P2 diagnostic probes. No later stage is computed speculatively.
