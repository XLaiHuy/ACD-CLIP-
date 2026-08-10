# P1-v8.4-A Fingerprinted Router Baseline Decision

The new forward-only baseline is authoritative for Router development. It consumed exactly 300 fingerprinted VisA/train samples from the regenerated checkpoint, with manifest SHA256 `20575581d04b90b1221130d9870e5a6b50466d942c88eeb8bf18c39c474cc25b`.

All model-state, gradient-absence, residual, routed, ActualGated, and batch-count invariants passed. Margin eligibility is structurally healthy: overall 99.428080%, normal 99.592109%, anomaly 71.553229%. Relative to legacy context, anomaly is identical and normal/overall differ by one selected patch. Anomaly eligible winners remain F2=2,215, F3=1,895, F4=400.

Router q targets are nevertheless nearly uniform on the margin-selected set. Median normalized entropy exceeds the predeclared 0.98 reference for both overall and anomaly at tau=.05, .03, and .02. Consequently the correct decision is `ROUTER_TARGET_FORMULATION_UNRESOLVED`.

No Router lambda calibration, Router 8B smoke, tau change, threshold change, loss reweighting, balancing, capacity change, or Router 300B launch is authorized. The next action is discussion of the Router target formulation; the fingerprinted baseline remains the authoritative evidence for that discussion.

EXIT_FOR_DISCUSSION
