# Resource/reporting record

This archive is compact and contains reports, CSV summaries, JSON manifests, and audit code references only. It intentionally excludes raw per-pixel stores, memmaps, evaluator spools, caches, huge logs, and checkpoints.

Completed replay work used one temporary historical worktree and bounded current-evaluator replay outputs under runs/phase2b_historical_gain_forensics_20260901/. Temporary replay spools were cleaned after each target. No training process was running during the final audit check, no duplicate training process was launched, and no new training result is claimed.

The C2 E10 serialization incident is documented separately; its consequence is a model-state-only comparison, not a resource failure or a reason to fabricate a replacement checkpoint.
