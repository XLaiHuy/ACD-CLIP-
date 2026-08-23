# SABRA-CAR R1 Recovery Protocol v1

Status: FROZEN BEFORE RECOVERY IMPLEMENTATION OR EXECUTION

## Protocol identity and scope

This document is a new, explicit post-failure recovery preregistration. It is
distinct from, and does not amend, replace, reinterpret, or rewrite, the
immutable original S0 protocol.

- original S0 SHA: 08ca99ff69d6d85184f5d145830876befb413628
- pre-recovery workflow SHA: 719a14f8415a939322e90ccb598cc90c178015eb
- original failure artifact: results/sabra_car/r1/FIT_FAILED.json
- original failing fold: held-out candle
- original solver event: LBFGS reached max_iter=1000
- original R1 predictions produced: no
- original R1 scientific metrics produced: no
- original R1 scientific gate evaluated: no
- MVTec accessed: no
- Medical accessed: no

The immutable original attempt remains recorded as:

R1_ORIGINAL_STATUS=INCONCLUSIVE_SOLVER_FAILURE

It is not a scientific FAIL, scientific PASS, threshold STOP, or evidence
against the R1 hypothesis. No prediction, metric, threshold decision, or R1
scientific gate value existed. The recovery protocol is frozen before recovery
implementation, before recovery execution, and before observing any R1 metric.

## Sole authorized computational change

Exactly one computational change is authorized:

- solver = lbfgs (unchanged)
- original max_iter = 1000
- recovery max_iter = 5000

The reason is computational only: allow the exact same frozen optimization
objective additional iterations to reach its existing convergence condition.
This is not permission to alter the scientific model, objective, data,
predictions, metrics, thresholds, or gate.

## Everything else remains frozen

Except for max_iter=5000, recovery must use the original S0 R1 definition and
the already-frozen R1 implementation contract without alteration:

- same dataset: the already-authorized VisA source data only;
- same allowed split: twelve-fold leave-one-class-out VisA predictions only;
- same twelve LOCO folds and held-out-class membership;
- same fold ordering, beginning with the original candle fold;
- same class-inventory, cache-image, and ascending patch-index sample ordering;
- same eleven features in the same frozen order;
- same certified three-class source-oracle action target definition;
- same training-fold-only median/IQR preprocessing;
- same NumPy linear quantiles, IQR floor 1e-6, and normalization;
- same Phase2B E10 checkpoint and SHA256 identity;
- same immutable canonical and Trust-v2 GT-free caches and provenance checks;
- same random_state=0;
- same solver=lbfgs and multinomial logistic objective;
- same C=1.0, L2 regularization, and intercept treatment;
- same tol=1e-4;
- same initialization and no warm start or optimizer-state resume;
- same class_weight=balanced, all eligible patches, and no subsampling or class
  truncation;
- same estimator class order [-1,0,1], stable argmax, confidence, abstention,
  and KEEP definitions;
- same canonical prediction and signed-correction deployment definition;
- same pAP, pAUROC, coverage, opposite-sign-risk, relative-risk-reduction, and
  breadth metrics;
- same threshold grid {0.50,0.60,0.70,0.80,0.90} and the same selection rule;
- same R0-selected signed direction and fixed strength
  delta = predicted_action * 0.25 * 19.840438842773438;
- same original R1 pass/fail gate and tie rules;
- same continuation rule: R2 is authorized only after the complete original R1
  scientific gate passes;
- same deterministic, correctness, provenance, finite-value, zero-training,
  and zero-Medical-access requirements; and
- same fitting and evaluation environments and package versions frozen by the
  R1 implementation contract.

The original R1 scientific gate remains exactly:

1. action coverage at least 10%;
2. opposite-sign error among acted patches at most 5%;
3. opposite-sign error at least 25% relatively lower than the unfiltered
   non-KEEP argmax predictor;
4. LOCO CAR macro pAP improvement over native at least +0.50 pp;
5. macro pAUROC decline versus native no worse than 0.50 pp; and
6. at least 7/12 classes with non-negative pAP delta.

No recovery output may change any of these quantities or rules.

## Recovery convergence and decision rule

For every required R1 LOCO fit, SUCCESSFUL_FIT requires all of:

1. the same LBFGS objective used by original R1;
2. convergence before or at max_iter=5000 according to the existing solver
   convergence condition;
3. finite fitted parameters;
4. finite predictions; and
5. no implementation or infrastructure exception.

Only if all twelve required folds are SUCCESSFUL_FIT may the complete held-out
predictions be produced and the original frozen R1 scientific gate be evaluated.

If any required fold reaches max_iter=5000 without satisfying the frozen
convergence condition, then:

R1_RECOVERY_STATUS=COMPUTATIONAL_STOP

At that point:

- do not alter max_iter again;
- do not modify tolerance;
- do not change solver;
- do not attempt another recovery;
- do not evaluate an incomplete R1 scientific gate; and
- do not start R2, R3, or R4.

There is exactly one recovery attempt under SABRA-CAR R1 Recovery Protocol v1.
The attempt covers the complete fixed LOCO fitter; an aborted, non-convergent,
or exception-terminated required fold consumes that attempt. No per-fold retry,
resume, warm start, or replacement fit is authorized.

## Explicit prohibitions

This protocol forbids:

- changing tolerance;
- changing solver;
- changing regularization;
- changing initialization or resuming optimizer state;
- changing seed;
- changing folds or fold order;
- changing sample order;
- changing features or feature order;
- changing labels or target definition;
- changing preprocessing or normalization;
- changing prediction definitions;
- changing thresholds or threshold selection;
- changing scientific gate logic;
- changing any metric;
- changing directionality or signed strength;
- inspecting MVTec;
- inspecting Medical; and
- adding any further fallback after observing recovery results.

## Scientific interpretation

Any recovery execution is reported as an
AMENDED_PREREGISTERED_RECOVERY result. It must never be represented as though
max_iter=5000 were part of original S0. The original attempt permanently
remains INCONCLUSIVE_SOLVER_FAILURE.

If and only if the recovery converges for every required fold and produces all
required predictions, the unaltered original R1 hypothesis and gate may be
evaluated. No threshold, metric, model, direction, or gate may be changed based
on recovery outputs.

## Data firewall

At preregistration creation:

- MVTec_ACCESSED=NO
- MEDICAL_ACCESSED=NO

Recovery remains entirely within the dataset and split already authorized for
R1 by S0. MVTec and Medical remain forbidden before a protocol-valid downstream
authorization. This preregistration itself authorizes neither implementation
nor execution; both require explicit user approval after this document is
committed and published.

## No result-dependent choice

This protocol has one fixed computational change, one attempt, and one
deterministic computational-stop rule. It contains no branch that permits a
parameter, model, threshold, metric, gate, dataset, or fallback to be chosen
after seeing recovery behavior or results.
