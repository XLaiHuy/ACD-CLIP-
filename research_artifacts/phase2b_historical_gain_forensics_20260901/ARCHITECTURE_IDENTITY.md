# Architecture identity

H1 is historical V3c: hard-prompt Phase1, DFG attention/SS2D weight residual, FP32 attention residual flag, E9 checkpoint 6c1d888af56d011f7d2dabee7a5662ff422420df428841582a51c02846500e4a. H2 is historical hybrid Phase2B with alpha schedule and K-reg, checkpoint ae27443f99020588298a9ecc6dfc833a83ebe7a752f00e8524042d5a84a2c0cb. C2 is current canonical hybrid Phase2B with no K-reg, config SHA d24cf942684b0be3c12838699ec6fe452697bd7f0a58eabbf316fb79b1b18cdb, architecture freeze SHA f6de6ee8f1998f591c077efeff50fa9741a9f8bad34603ba145ec54ef961ba86.

The base CLIP/adapter/DFG dimensions are compatible. H1 and H2 are not identical prompt/loss contracts; H2 and C2 are not identical objectives/precision/evaluator contracts. Legacy H1/H2 checkpoints load through the current bridge only with an explicitly recorded legacy-metadata bypass. No architecture equivalence claim is made beyond the measured model-state replay.
