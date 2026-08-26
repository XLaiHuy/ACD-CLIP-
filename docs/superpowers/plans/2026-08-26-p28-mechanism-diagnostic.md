# P28 mechanism diagnostic implementation plan

## Scope

Implement one post-hoc, zero-training diagnostic on the frozen P27 artifacts.
The diagnostic will use only cached Tier-A logits/features, immutable N/S
prediction maps, frozen adapter checkpoints for alignment, and post-freeze
VisA held masks. It will not modify P27 scientific code or create P29 work.

## Sequence

1. Add synthetic tests for exact R0 action/margin semantics, region pooling and
   reconstruction, immutable N/S loading, rank metrics, pair-order accounting,
   and the MVTec/Medical/no-forward/no-optimizer firewall.
2. Implement a bounded, class-at-a-time P28 driver with no optimizer or model
   training path. Reuse existing frozen deployment and adapter modules.
3. Run synthetic tests, then static/pre-audit checks, inspect the explicit
   diff, and commit the P28 execution base.
4. Create one attempt marker containing execution-base, preregistration,
   artifact-manifest, and asset hashes. Run the complete 12-class diagnostic
   exactly once.
5. Generate the required metrics, ranking, alignment, root-cause, report, and
   post-run audit artifacts; commit only small terminal evidence; push and
   verify remote equality and a clean worktree.

## Checkpoints

- P28 preregistration commit is already pushed before implementation.
- No held mask is read before the single attempt marker.
- No diagnostic outcome is used to alter code, protocol, or execution.
- P28 terminal state is either `P28_DIAGNOSTIC_COMPLETE` or
  `P28_ENGINEERING_STOP`.
