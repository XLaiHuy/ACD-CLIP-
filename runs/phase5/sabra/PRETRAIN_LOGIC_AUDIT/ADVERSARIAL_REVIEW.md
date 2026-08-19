# SABRA adversarial review

- GT-free cache was finalized and hash-checked before mask reads.
- Peer selection uses only Phase2B ranks/features; GT is post-hoc only.
- Perturbations use the baseline image CDF and never rerank independently.
- Need is OOF LOCO and uses the four frozen base features only.
- Oracle parity uses the frozen central finite difference sample.
- Statistical unit is VisA class; 10,000 class bootstraps and exact 4096 sign flips are retained.
- MVTec, medical data, and Phase2B training were not accessed.
- Stable-but-wrong and contaminated-reference vetoes were explicitly evaluated.
