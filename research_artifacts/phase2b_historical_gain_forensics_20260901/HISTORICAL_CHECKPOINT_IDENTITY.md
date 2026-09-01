# Historical checkpoint identity

H1 is the V3c FP32-attention E9 checkpoint 6c1d888af56d011f7d2dabee7a5662ff422420df428841582a51c02846500e4a (56426187 bytes) from 0232200e964c02c328eec09dbe842f327f72fcd9. H2 is the exact Phase2B hybrid/K-reg E10 checkpoint ae27443f99020588298a9ecc6dfc833a83ebe7a752f00e8524042d5a84a2c0cb (56452037 bytes) from e03966997d4cecfd985943a4053a93e1e40197ec. Both are legacy model-state payloads and do not contain optimizer, scheduler, or RNG state.

The C2 parent E10 provenance originally recorded SHA 31ca8344c646693d0ee51941d39f28aa07b6a102c49d1efdc5e3cdf2ec8bcc50 and full optimizer/scheduler/RNG metadata. During a temporary serialization-only replay preparation, a symlink to that file was passed to torch.save; the save followed the symlink and overwrote the physical file with a stripped model-state payload. The model tensors were preserved, and the original SHA plus all original compact metric records remain in the frozen corrective/extension archives, but the original full checkpoint metadata is not recoverable from this workspace. Therefore all C2 E10 replay claims are explicitly MODEL_STATE_REPLAY_ONLY.

The candidate table is HISTORICAL_CHECKPOINT_CANDIDATES.csv.
