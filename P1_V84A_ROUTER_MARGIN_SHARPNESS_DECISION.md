# P1-v8.4-A Frozen Router Margin Sharpness Replay Decision

This is the one authorized actual forward-only replay for the missing Router q-target sharpness evidence. It used the provenance-approved regenerated checkpoint (`96f679b2e18f4e352157494f7198414b66f66024a5cc023f5ff046c39dcaa3a3`), OpenAI CLIP artifact, seed 0, VisA/train, image size 518, FP32, AMP/TF32 off, and 300 microbatches.

The historical canonical data root is `/workspace/data/med_visa/data`. Its 2,162 VisA manifest rows have no missing required files. An initial setup process inherited the repository fallback root and failed before fetching its first batch; it produced no model forward or output artifact. The single actual replay then used the canonical root and reached 300/300 batches.

The replay failed only the mandatory invariant `reconstructs_historical_margin_support`. Its own invariant failure report consequently establishes that gradients remained absent, model state remained unchanged, residual/routed/ActualGated reconstruction remained exact, and exactly 300 batches ran. The checkpoint SHA256 was independently unchanged after the process exited.

The audit deliberately refused to persist q-sharpness aggregates after the support reconstruction mismatch. Therefore no canonical tau decision is scientifically justified. No lambda calibration, Router 8B smoke, Router 300B, tau change, threshold change, loss reweighting, capacity change, or ACT/factor change was run.

Decision: `FROZEN_ROUTER_AUDIT_REPLAY_INVALID`.

The next action requires discussion of the frozen-support mismatch and authorization for any future diagnostic attempt. It must not be treated as a tau recalibration result.

EXIT_FOR_DISCUSSION
