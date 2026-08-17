# P5-FR1 MVTec geometry contract recovery

## Terminal

`P5FR1_AUDIT_INVALID`. The prior P5-F terminal remains `P5F_AUDIT_INVALID`.

The single authorized GT-free process finalized 750 of 1,725 records and then returned with canonical identity `750` still marked `INFLIGHT`. The command status did not persist a completion or exit code. The exact process termination cause is unavailable. Because the forward status for that identity is uncertain, the recovery protocol forbids a resume or rerun.

No all-configuration evidence was computed after the invalidation. No GT, masks, model reload, post-hoc evaluator, candidate triage, or scientific metric was run. The partial cache is preserved at `/tmp/p5fr1_mvtec_common` as local forensic state and is not treated as a valid scientific pass.

`G0=FAIL`; `G1`--`G4` are not reached; `candidate=NONE`; `E1_AUTHORIZED=false`.

The prior P5F worktree and terminal remain immutable.
