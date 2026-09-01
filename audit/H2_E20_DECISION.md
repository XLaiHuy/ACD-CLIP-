# H2 E20 decision

Decision: `FOUND_DIFFERENT_RUN`; the exact H2 horizon remains E15.

The exact hash-identified H2 run has `adapter_1.pth` through
`adapter_15.pth` and no E16-E20 files. The E20 search found H2-shaped
artifacts, but none is the same scientific trajectory: the two base-repo
model-only E20 runs change `hybrid_alpha` and/or `lambda_k`, while the CIR,
Anchor, canonical-SABRA, and lab20e artifacts use different commits and/or
full-state intervention contracts. Their hashes and metadata are tabulated in
`H2_E20_SEARCH.csv`; model-only artifacts also do not carry complete optimizer,
AMP, RNG, manifest, or CLIP provenance.

The search also found unrelated Phase1, Phase3, v2-phase2, and medical-test
E20 families. They are excluded from the H2-like table because their
research-line or architecture identity is different; none can extend the
exact H2 trajectory.

Therefore tomorrow's clean factorial is preregistered at fixed confirmatory
epoch E15. E10 may be retained for diagnostics only. No Medical or MVTec
result from any E20 artifact is used to select that horizon, and no
performance winner is claimed.
