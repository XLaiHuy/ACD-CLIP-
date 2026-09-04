# H2 redesign gate

## Gate inputs

| requirement | current evidence | gate result |
|---|---|---|
| Stable replication | Seed 1 and Seed 2 both have nonfinite-gradient skips and unequal H/A successful steps | `FAIL` |
| Protocol mismatch ruled out | Internal H2 contract passes, but public protocol is only partial | `FAIL` |
| Evaluator mismatch ruled out | Internal oracle parity passes; public evaluator details remain unknown | `PARTIAL` |
| Horizon explanation insufficient | Seed-0 E15/E20 is metric-specific, but replication is invalid | `NOT_ESTABLISHED` |
| LR/optimizer/loss-scale failure insufficient | Numerical mechanism is not instrumented enough to localize | `NOT_ESTABLISHED` |
| Initialization variance insufficient | No valid Seed-1/Seed-2 target variance exists | `FAIL` |
| Repeatable mechanistic residual signature | Only aggregate Seed-0 metrics and source telemetry exist | `FAIL` |
| Current architecture lacks the required mechanism | Not testable before the preceding gates pass | `NOT_ESTABLISHED` |

`REDESIGN_AUTHORIZED=NO`.

No architecture search or literature-driven novelty is authorized. The
failure is upstream of target comparison: replication validity failed before
Medical/MVTec evaluation. The safe next step is a single source-only
numerical-stability investigation followed by the smallest implementation
repair if a concrete operand/branch is identified.

`FINAL_ACTION=OPTIMIZATION_REPAIR`.
`NEXT_FULL_TRAIN_AUTHORIZED=NO`.
`TARGET_LABELS_USED_FOR_SELECTION=NO`.

## Red-team review

`ROOT_CAUSE_REDTEAM=PASS` means the leading explanation is restricted to what
the evidence can actually support: repeated numerical invalidity is a strong
validity blocker, not a proven explanation of the Medical AP or MVTec AUROC
gap. The following falsifications remain active:

1. The apparent seed effect could be artifact noise because no valid target
   metrics exist for Seeds 1 and 2.
2. The external gap could partly be evaluator/protocol mismatch because stride,
   rounding, and public implementation details are not identical.
3. The heterogeneous Seed-0 class pattern could be ordinary per-class noise;
   it is not evidence of a single morphology mechanism.
4. Correlation between prompt/DFG telemetry and the target gap cannot be
   causal without target score maps or feature traces.
5. Adding capacity or a fashionable module would not address the currently
   demonstrated hard blocker and could violate zero-shot comparability.

## Required next gate sequence

1. Preserve the invalid logs/checkpoints as forensic evidence.
2. Instrument or reproduce the source-only numerical failure without changing
   scientific settings.
3. Apply at most one minimal stability repair after the cause is demonstrated.
4. Re-run the frozen validity gate for fresh H/A Seeds 1 and 2.
5. Only a valid replication can reopen target evaluation and the redesign gate.
