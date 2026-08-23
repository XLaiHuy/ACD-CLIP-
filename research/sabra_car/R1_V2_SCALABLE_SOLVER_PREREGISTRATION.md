# SABRA-CAR R1 Scalable Solver Protocol v2

Status: FROZEN BEFORE R1-v2 IMPLEMENTATION OR EXECUTION

## Protocol identity and historical boundary

This is a new preregistration derived from terminal R1-v1 HEAD
`782b8b81aa5b03c88a4417e5d7106e19ceff83ce`. It is distinct from and does not
amend, replace, reinterpret, or rewrite:

- original S0 SHA `08ca99ff69d6d85184f5d145830876befb413628`;
- R1 Recovery v1 preregistration SHA
  `6f8fed381838a0221bb5289fb23d2243a6b5f0ef`;
- original R1 failure evidence at `results/sabra_car/r1/FIT_FAILED.json`; or
- terminal R1-v1 evidence at
  `results/sabra_car/r1_recovery_v1/COMPUTATIONAL_STOP.json`.

The original R1 attempt remains
`INCONCLUSIVE_SOLVER_FAILURE_BEFORE_SCIENTIFIC_RESULT`. R1 Recovery v1 remains
`COMPUTATIONAL_STOP`. Neither attempt produced complete LOCO predictions, R1
scientific metrics, or an R1 scientific gate result. Therefore the v2 solver
choice is frozen before any R1 scientific outcome has been observed.

## Frozen computational evidence

For the first LOCO fold, holding out `candle`:

- training matrix shape: 2,685,978 rows by 11 features;
- held-out rows: 273,800;
- training targets: SUPPRESS 1,316,489; KEEP 1,255,982; BOOST 113,507;
- LBFGS Recovery v1 reached exactly `max_iter=5000`;
- candle fit elapsed time: 2370.1823143600486 seconds;
- no OOM, NaN/Inf, or implementation exception was observed; and
- no prediction, metric, or scientific gate value was produced.

## Compatibility verification before freeze

The installed frozen fitter environment was inspected without fitting:

- scikit-learn version: 1.7.2;
- `newton-cholesky` is documented as supporting multinomial multiclass loss;
- `newton-cholesky` is documented as supporting L2 regularization;
- `class_weight="balanced"` is accepted by the exact estimator configuration;
- estimator parameter validation: PASS; and
- scientific or timing fit executed during compatibility verification: no.

No substitute solver is authorized if this exact compatibility contract later
fails in the frozen execution environment.

## Sole solver change

The scientific model remains multinomial logistic regression. The sole changed
computational component is the optimization solver:

- old solver: `lbfgs`;
- new solver: `newton-cholesky`.

The exact R1-v2 estimator is frozen as:

- `solver="newton-cholesky"`;
- `max_iter=100`;
- `tol=1e-4`;
- `C=1.0`;
- `penalty="l2"`;
- `class_weight="balanced"`;
- `random_state=0`;
- `fit_intercept=True`;
- `multi_class="multinomial"`;
- `warm_start=False`; and
- primal formulation (`dual=False`).

The finite cap `max_iter=100` is frozen before implementation and before any
R1-v2 fit. It is the documented scikit-learn 1.7.2 `LogisticRegression`
default applicable to `newton-cholesky` and is intentionally not copied from
LBFGS's 5000-iteration budget. It is a solver-specific computational stop
boundary, not permission to change the model or convergence tolerance.

## Scientific rationale

The frozen candle training fold has approximately 2.686 million rows but only
11 features and 3 classes. `newton-cholesky` is selected for this extreme
`n_samples >> n_features * n_classes` geometry. Its explicit Hessian has
quadratic memory dependence on `n_features * n_classes`, which is bounded here
by the very small feature/class dimension, while the multinomial
logistic-regression objective remains unchanged.

This rationale authorizes no post-result solver selection, comparison ladder,
or fallback.

## Everything else remains frozen

R1-v2 must preserve exactly:

1. VisA source data and its authorized role;
2. all twelve leave-one-class-out folds and held-out-class membership;
3. fold order beginning with `candle`;
4. class-inventory order, cache-image order, and ascending patch-index order;
5. every eligible patch, with no subsampling or class truncation;
6. three-class BOOST, KEEP, and SUPPRESS R0 oracle targets;
7. the exact eleven-feature order:
   - `margin_within_image_rank`;
   - `robust_margin_normalization`;
   - `D_rank`;
   - `deployment_sensitivity`;
   - `E`;
   - `peer_coherence`;
   - `query_support_mean`;
   - `peer_eigen_entropy`;
   - `stage_query_profile_disagreement`;
   - `supported_p9_stability`;
   - `supported_p16_stability`;
8. training-fold-only median/IQR preprocessing;
9. NumPy linear quantiles and IQR floor `1e-6`;
10. the exact Phase2B E10 checkpoint, cache identities, and provenance checks;
11. the frozen estimator quantities listed above other than the solver change;
12. estimator class order `[-1,0,1]`;
13. stable argmax and confidence as maximum class probability;
14. threshold grid `{0.50,0.60,0.70,0.80,0.90}`;
15. below-threshold prediction as KEEP;
16. action coverage, opposite-sign error, unfiltered comparator, relative-risk
    reduction, and threshold-selection definitions;
17. R0 fixed signed intervention strength
    `delta = predicted_action * 0.25 * 19.840438842773438`;
18. canonical exact pAP and pAUROC implementation and per-class breadth;
19. every original R1 scientific gate and tie rule;
20. continuation to R2 only after a complete original R1 gate PASS;
21. deterministic, finite-value, probability-sum, correctness, and provenance
    requirements; and
22. zero Phase2B training steps and no Phase2B parameter updates.

## Convergence and computational-stop rule

For every required LOCO fold, `R1_V2_SUCCESSFUL_FIT` requires:

1. the exact frozen multinomial/L2/balanced objective above;
2. convergence before or at `max_iter=100` according to the existing
   scikit-learn `newton-cholesky` convergence condition;
3. estimator classes exactly `[-1,0,1]`;
4. finite coefficients and intercepts;
5. finite held-out probabilities whose rows sum to one within `1e-6`; and
6. no implementation or infrastructure exception.

If any required fold reaches `max_iter=100` without satisfying the frozen
convergence condition:

`R1_V2_STATUS=COMPUTATIONAL_STOP`

Then preserve the evidence and stop. Do not increase `max_iter`, change
tolerance, change solver, alter preprocessing, retry with another
initialization, evaluate an incomplete scientific gate, or start R2-R4.

There is one complete R1-v2 attempt under this protocol. No solver fallback or
additional recovery strategy is authorized inside v2.

## Scientific evaluation rule

Only if all twelve folds are `R1_V2_SUCCESSFUL_FIT` may complete held-out
predictions be assembled and the unchanged original R1 procedure be evaluated.
R1 passes only if all correctness/provenance gates pass and:

- action coverage is at least 10%;
- opposite-sign error among acted patches is at most 5%;
- opposite-sign error is reduced at least 25% relatively versus the unfiltered
  non-KEEP argmax predictor;
- LOCO CAR macro pAP improves over native by at least +0.50 pp;
- macro pAUROC decline versus native is no worse than -0.50 pp; and
- at least 7/12 classes have non-negative pAP delta.

If no frozen threshold qualifies or any complete original scientific gate
fails, R1-v2 is `SCIENTIFIC_STOP`. Such a result must not be relabeled as an
engineering or computational failure.

## Explicit prohibitions and data firewall

R1-v2 forbids changing any frozen data, target, feature, preprocessing,
normalization, fold, ordering, sample, estimator parameter other than the
preregistered solver replacement, threshold, metric, intervention, gate, or
continuation rule. It also forbids subsampling, class truncation, alternate
optimization, neural fallback, and any result-dependent choice.

- `PHASE2B_TRAINING_STEPS=0`
- `MVTec_ACCESSED=NO`
- `MEDICAL_ACCESSED=NO`

MVTec and Medical may not be enumerated, inspected, hashed, loaded, evaluated,
or otherwise accessed during R1-v2 implementation or execution. R2-R4 are not
authorized by this preregistration task.

## Authorization boundary

This document authorizes preregistration publication only. R1-v2 implementation
and execution require subsequent explicit user approval after this file is
committed and pushed. No R1-v2 fit has been run.
