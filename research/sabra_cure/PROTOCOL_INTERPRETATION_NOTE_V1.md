# SABRA-CURE Protocol Interpretation Note v1

Status: prospective clarification before implementation and any SABRA-CURE R1 result.

This note does not amend `MASTER_PREREGISTRATION_V1.md`. It resolves only the
following textual interpretation:

- An R1 failure is terminal and forbids R2.
- An R2 failure is terminal and forbids R3 and R4.
- An R3 failure of the utility-magnitude proposal is a valid negative R3
  result. The already-certified fixed-strength R2 mechanism may remain eligible
  for R4 exactly as Section 8 states, only when the complete R2 gate still
  holds.
- Section 8 controls that specific R3 fallback semantic.

The conservative-utility phrase “interval crosses zero” is operationalized
only by the frozen rule `r = abs(mu) / sigma`: action occurs only when `r > k`
for the preregistered `k` grid. This note does not introduce an alternative
interval formula.

No formula, feature, threshold, model, metric, gate, dataset role, or stop
criterion changes. No scientific R1, R2, R3, or R4 result existed when this
note was made.
