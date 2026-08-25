# P26 External Validation Contract

`EXTERNAL_VALIDATION_AUTHORIZED = FALSE`

P26 does not access or evaluate MVTec or Medical. After explicit user review,
one separately authorized untouched MVTec evaluation may use exactly the P26
architecture commit and config. It may not tune thresholds, alpha, prompts,
preprocessing, checkpoint, action policy, or architecture from MVTec results.
Medical remains forbidden.

The external run must verify the architecture commit, config hash, checkpoint
hash, CLIP asset hash, clean worktree, and authorization artifact before any
dataset read. No architecture revision is allowed after the external result.
