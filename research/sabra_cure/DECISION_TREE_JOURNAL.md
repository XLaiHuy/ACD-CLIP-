# SABRA-CURE Phase-0 Decision Journal

## Deterministic adversarial review

Exactly 100 desk-review stress tests were performed: 20 risk dimensions × 5 questions. Dimensions were novelty, performance plausibility, adaptability, source overfitting, class generalization, signed-risk behavior, calibration, patch dependence, spatial correlation, label leakage, target leakage, feature degeneracy, model capacity, class imbalance, utility imbalance, distribution shift, runtime, memory, implementation complexity, and reproducibility. For each dimension the questions were: strongest failure argument; whether existing evidence refutes it; possible simplification; new confounder introduced; and whether repair would require seeing results. These were research reviews, not training runs or empirical results.

Top objections: millions of correlated patches may exaggerate effective sample size; source-LOCO uncertainty may shift by class/domain; uncertainty may learn class identity; bounded utility may obscure extreme gains; clustered harmful actions can survive patch-level gates; and the local derivative is not a causal potential outcome.

Accepted repairs: class-level LOCO only; inner class cross-fitting for residual targets; two linear closed-form heads; training-fold-only scaling; per-class breadth gates; explicit spatial diagnostics; fixed finite risk grid; bounded reversible correction; exact native parity at zero; no conformal or causal guarantee claim.

Rejected ideas: calibration-only main method; hierarchical classifiers; high-capacity neural fallback; separate BOOST/SUPPRESS value surfaces; arbitrary feature accumulation; target-domain calibration; radius classifier; hyperparameter sweep; second CLIP forward; and result-dependent recovery.

Unresolved risks remain scientific stop opportunities. None authorizes tuning after results.

## Decision nodes

- **D0 — R1 diagnosis:** confidence is class-dependent and globally associated with more opposite-sign error; H0 remains comparator only.
- **D1 — utility:** valid, finite, aligned, source-only; T2 is stable.
- **D2 — signed features:** three minimal cache-only signed features survive; no signed-information bottleneck.
- **D3 — formulation:** H3 survives. H2 is the required ablation; H1/H4 are unsupported complexity.
- **D4 — risk:** use uncertainty-normalized conservative utility, selected independently of pAP.
- **D5 — radius:** test direct bounded utility magnitude first; no radius classifier.
- **D6 — final:** `PASS_PREREGISTERED`.

Surviving formulation: source-LOCO two-stage ridge prediction of transformed signed utility and log cross-fitted absolute residual, with conservative uncertainty-normalized abstention and a separately gated bounded correction-strength stage.
