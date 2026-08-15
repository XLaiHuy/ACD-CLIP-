# Evidence to design

## Frozen contract

Use only the frozen GT-free quantities: `m_bar`, `D_rank`, `E_nonlocal`, `valid_reference`, same-image peer indices/features, and other native Phase2B outputs. `D_rank` is ranking risk, not anomaly evidence. `E_nonlocal` is auxiliary same-image reference evidence. GT is allowed only to label archived outcomes after prediction/action freeze.

The exact deployment path is:

```text
native 37x37 correction -> Gaussian blur 7x7 sigma1 -> bilinear resize align_corners=True -> stage mean -> softmax
```

## Committed evidence

| Evidence | Exact result | Design implication |
|---|---:|---|
| Corrected B2 matched bridge | aligned `W=0.676282...`; 11/12 supportive classes; aligned beats shifted | E carries useful pairwise information, but the bridge is a GT-defined evaluation population, not an inference selector. |
| B3/B3.1 parity | 2,785 archived pairs; aligned 592 rescued, 202 broken, net +390; shifted 534 rescued, 847 broken, net -313 | The action can help and harm; shifted E is a control, not a deployable rule. |
| Base rank geometry | aligned gap >10 contains 461/592 rescues and net +308; gap 1 has 18 rescues and net +14 | Candidate A is not justified; useful evidence is not adjacency-concentrated. |
| Class bootstrap | gap >10 net CI `[5.0833,49.9167]` across all 12 classes; gap 4–5 CI `[0.0833,3.0]`; gaps 2 and 3 cross zero | Broad-gap utility is class-consistent as a diagnostic, but does not define a selector or causal mechanism. |
| Spatial relation | rank-gap vs Chebyshev Pearson `-0.0134`, Spearman `-0.0120`; rescued and broken distances overlap and are broad | Rank-locality is not spatial-locality; future action needs deployment-aware spatial support, but no distance threshold is supported. |
| Deployment comparison | native aligned-minus-shifted AP macro `+0.005357`; after deployment `-0.001311`; reversal in 7/12 classes | Native ordering improvement is not deployment improvement; future candidate must be tested end-to-end. |
| Action magnitude | B3 class-bootstrap relation between displacement and breakage `[0.7027,0.9539]` | A bounded/trust-region family is mechanistically motivated. |
| Existing action | C1 sorts all eligible slots by E within cells | This is broad E reranking and is closed; it cannot be reused as the missing pair proposal. |
| Existing pair rows | `bridge_matches` uses post-hoc GT labels and B3 emits rows after labels are loaded | B3/B3.1 rows cannot become inference constraints. |

## Gate 3 audit

Gate 3 requires a concrete GT-free contract for pair proposal, E acceptance, ambiguous abstention, repeated/conflicting constraint prevention, unrelated-order preservation, and bounded native authority/spatial support. The persisted evidence does not contain an already-supported rule satisfying all items. The B3 bridge gives outcome labels only after using GT. B2/C1 supplies cell membership and a full E-sorted assignment, but not a disjoint, abstaining, bounded pair matcher. Creating such a matcher now would be a new method rule.

## Candidate-family comparison

- **A / selective adjacent one-step:** rejected. Positive aligned utility is not concentrated at gap 1 or gaps 1–3; gap >10 dominates rescues and net utility.
- **B / symmetric or positive-only minimum projection:** mechanistically compatible with broad evidence and displacement harm, but not frozen because the required GT-free proposal and acceptance/abstention/conflict contract is missing. P5A’s family-B label is not a deployable selector.
- **G / broader partial-order projection:** rejected. It risks cascading and C1-like broad movement; no minimum-distortion plus deployment-safety guarantee is present.

## Decision

`selected_candidate = NONE`, `terminal = P5B_RELIABILITY_AUDIT_REQUIRED`. This is a reliability/design-contract blocker, not a claim that pairwise evidence is useless. No implementation, tuning, full candidate evaluation, or medical-transfer claim is authorized.
