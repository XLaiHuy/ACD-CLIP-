# SABRA Trust-v2 adversarial review

- GT-free cache provenance, shard hashes, frozen assets, and pushed implementation were checked before VisA masks were opened.
- Peer selection and p9/p16 reserves were computed without GT; GT was used only for post-freeze evaluation.
- Models are class-held-out OOF balanced logistic regressions with training-fold-only scaling.
- MVTec and medical data were not accessed.
- PCRR is diagnostic and never fused into the primary Trust-v2 score.
