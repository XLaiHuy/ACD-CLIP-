# P5 B3.1 action geometry diagnostic

Status: PASS. Measurement-only; no P5 candidate was implemented.

## Protocol and integrity

- Successful full pass: 2,162 canonical VisA TEST image forwards; 0 training steps.
- Raw pair records: 2,785; persisted per class, atomically reopened and validated before parity.
- Exact B3 parity: aligned 592 rescued / 202 broken / 1,417 preserved / 574 missed / net +390; shifted 534 / 847 / 772 / 632 / net -313.
- GT firewall: labels were attached only after frozen prediction, evidence, action, and deployment calculations.
- Three temporary plumbing restarts stopped after the first class; their partial rows were excluded. The results below are from the completed 2,162-forward pass.

## Base-rank geometry

| BASE gap | aligned n | rescued | broken | preserved | missed | net | rescue rate | break rate | shifted net |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42 | 18 | 4 | 16 | 4 | +14 | 81.8% | 20.0% | +3 |
| 2 | 42 | 12 | 10 | 13 | 7 | +2 | 63.2% | 43.5% | -11 |
| 3 | 47 | 14 | 8 | 21 | 4 | +6 | 77.8% | 27.6% | -5 |
| 4–5 | 67 | 27 | 11 | 23 | 6 | +16 | 81.8% | 32.4% | -7 |
| 6–10 | 201 | 60 | 16 | 81 | 44 | +44 | 57.7% | 16.5% | -7 |
| gt–10 | 2386 | 461 | 153 | 1263 | 509 | +308 | 47.5% | 10.8% | -286 |

Aligned utility is not adjacent-local: gap 1 contributes +14 net, while gap >10 contributes +308 net and contains 461/592 aligned rescues; gap 6–10 contributes +44. Class-bootstrap net CIs are positive for gap >10 ([5.08, 49.92]) and gap 4–5 ([0.08, 3.00]), while gap 2 and gap 3 cross zero. This rejects adjacent-only A and supports bounded B.

Aligned score gaps overlap: rescued median 0.0995 (p95 0.7215), broken median 0.0607 (p95 0.6931). No near-tie threshold was invented or searched.

## Spatial geometry and deployment

Aligned rescued pairs have native Chebyshev mean/median 14.45/14.00; broken pairs 13.37/13.00. Shifted rescued/broken are 13.24/13.00 and 14.28/14.00. Rank gap versus Chebyshev distance is Pearson -0.0134 and Spearman -0.0112; versus Euclidean, -0.0126 and -0.0111. Rank-locality is not spatial-locality, and no spatial cutoff is justified.

Native aligned-minus-shifted AP is positive in all 12 classes before deployment (macro mean +0.00536, 95% CI [+0.00307,+0.00819]) but -0.00131 after blur 7x7 sigma1 -> bilinear resize align_corners=True -> stage mean -> softmax (95% CI [-0.00249,-0.00026]); the deployed contrast reverses in 7/12 classes. This requires a spatial-support deployment guardrail without claiming causal isolation.

## Per-class consistency

Aligned net is positive in 10/12 classes; shifted net is positive in 0/12. Complete class strata and class-bootstrap CIs are in `PER_CLASS.json`.

## Design-family decision

**B — bounded minimal pairwise projection.**

A is rejected because useful aligned evidence is broad in BASE rank space rather than concentrated at adjacent ranks. C is rejected because broader partial-order movement risks C1-like permutation and this diagnostic supplies no minimum-distortion guarantee. Candidate B remains design-only: it must accept or abstain on GT-free E constraints, preserve unrelated Phase2B ordering, bound action magnitude, and validate the exact native-to-deployed path.

Next design-review question: can that projection enforce a native spatial-support/trust-region constraint without broad spatial score-mass movement?

Forbidden: threshold search, AP-driven tuning, training, a new VisA candidate, changes to K=8/top-20%/10x10/shift/pair rules, GT in inference/action, predictor/model edits, or reopening C1 and unrelated method families.
