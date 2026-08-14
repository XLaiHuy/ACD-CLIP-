OVERNIGHT DECISION: STAGE_RESCUE_ORACLE_ONLY

INPUT INTEGRITY: PASS
OUTPUT INTEGRITY: PASS

MECHANISM:
Branch A found real internal-stage rescue evidence and recovered 14.6901% of the A1 positive-only oracle AP gain on the class mean. Fixed-stage and GT-free P2/P3 arbitration failed to preserve it: P3 harmed AP in 12/12 classes and AUROC in 9/12. The rescue is therefore oracle-only, not a deployable selector.

Q1 POTENTIAL: medium
Q2 POTENTIAL: medium
EXTERNAL REPLICATION: unavailable; MVTec not installed

INFERENCE:
  completed full VisA TEST passes: 3
  Branch-A implementation-only first-image retries: 2
  successful branch runtime: 00:36:49
  progress polls: 68
  longest wait: 30 seconds

NEXT:
Audit candidate SECOND-EVIDENCE sources on identified held-out high-risk weak-positive pixels.
