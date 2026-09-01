# Strong-parent anchor reference decision

Status: BLOCKED_PENDING_RESTORED_PARENT_PARITY.

The current C2 P E14 reference must not be transplanted to H2 because H2 has a different prompt/loss/precision trajectory. H2 E10 is retrospective Medical-selected and cannot be used as a clean target-blind anchor reference. A scientifically admissible first reference could be a fixed H2 E1 model-state checkpoint under a preregistered rule, but compatibility and source-only behavior must be verified after restored-H2 parity. No anchor full train has been authorized in this snapshot.

Required frozen mechanism remains normalized per-parameter squared distance on image_adapter, frozen reference, lambda_image_anchor=0.001, no optimizer registration for the reference, and anchor absent at inference.
