# H2 BF16 Seed1 E15 numerical audit

The fresh `H2_BF16_SCREENING_SEED1_V1` source run completed before target
access. Its shared E1, H E15, and A E15 checkpoints are full-state BF16
checkpoints produced by commit `efdfc3747b100a56d91d9a0e04a7d420a863fcbf`.

| checkpoint | SHA-256 | steps | loss events | grad events | parameter state | Adam state |
|---|---|---:|---:|---:|---|---|
| shared E1 | `04b1aec8b24f1cc9e8ca76dedf8ba19aec086e789def3c8fda91949ca763d9ed` | 361 | 0 | 0 | finite | finite |
| H E15 | `f7cfbdc9835991dfdb0235673f039e12d1d5358f4f954da7b5783741a2a6ef43` | 5,415 | 0 | 0 | finite | finite |
| A E15 | `6848b06983a6c5bc4abadf5b6b67640d74ea8c7ed1bbd7ec4111fe7d23c44718` | 5,415 | 0 | 0 | finite | finite |

Every checkpoint records `precision=bf16`, `amp_enabled=true`,
`gradscaler_enabled=false`, `scaler_state={}`, and `tf32_enabled=false`.
The source E15 trajectories each attempted and successfully completed 5,415
optimizer steps. The immediate-abort numerical guard never fired.

At A E15, Safe Anchor remained finite and within its frozen family budget:
`rho=0.1`, `max_effective_active_family_ratio=0.09999997979333895`, and
`family_partition_complete=true`. Its final aggregate global task-gradient
norm was 0.6022872; the effective Anchor norm was 1.2597e-05. This supports
the intended capped, non-dominant Anchor behavior.

The target evaluators required a non-scientific PyTorch >=2.6 compatibility
change: the SHA-verified full-state checkpoints include NumPy/RNG metadata,
so `torch.load(..., weights_only=False)` is required. The external Medical
evaluator patch is preserved at
`/workspace/h2_bf16_screening/target_eval/medical/EVALUATOR_TORCH_LOAD_COMPAT.patch`.
The same compatibility fix is committed in `test.py` for MVTec. No model,
loss, optimizer, metric definition, checkpoint, or target setting was changed.
