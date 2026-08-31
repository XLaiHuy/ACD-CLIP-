# Next direction after the corrected diagnosis

Decision: PHASE2B_REPRESENTATION_PRESERVATION.

The required matched corrective retrain is complete. No new run is launched in this snapshot. If work resumes, run one source-only representation-preserving variant against native Phase2B, with the same seed, source, CLIP asset, FP32, batch, optimizer, schedule, checkpoint cadence, and evaluator. Keep operator consistency and RMT transport as separate identities.

The next gate is not a Medical-selected alpha. It is a held-out source or source-category transfer check plus checkpoint stability. If the preservation variable does not recover transfer without harming source Pixel AP, return to the native Phase2B representation. If a future RMT test is considered, first show usable, non-saturated peer evidence on source-only data; otherwise leave RMT out of inference.
