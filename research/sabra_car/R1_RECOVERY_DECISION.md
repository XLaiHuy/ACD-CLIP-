# SABRA-CAR R1 Recovery v1 Decision

Status: COMPUTATIONAL_STOP

## Immutable identities

- original S0 SHA: `08ca99ff69d6d85184f5d145830876befb413628`
- recovery preregistration SHA: `6f8fed381838a0221bb5289fb23d2243a6b5f0ef`
- recovery implementation SHA: `176179c95cd547f6ca19d7e9f5fdc157ce74e8b8`
- protocol: `SABRA-CAR R1 Recovery Protocol v1`
- authorized recovery attempts: 1
- recovery attempts consumed: 1

## Original R1 attempt

The original `max_iter=1000` attempt remains
`INCONCLUSIVE_SOLVER_FAILURE_BEFORE_SCIENTIFIC_RESULT`. It produced no complete
LOCO predictions, no R1 scientific metric, and no R1 gate result. Its evidence
remains unchanged at `results/sabra_car/r1/FIT_FAILED.json`.

## Recovery v1 result

The single preregistered recovery used the unchanged multinomial logistic
objective with `solver=lbfgs`, `C=1.0`, L2 regularization,
`class_weight=balanced`, `random_state=0`, and `tol=1e-4`. The sole changed
quantity was `max_iter`, from 1000 to 5000.

The first required LOCO fold, holding out `candle`, reached exactly 5000
iterations and emitted the LBFGS total-iteration-limit convergence warning.
Its fit elapsed time was 2370.1823143600486 seconds. The preregistered recovery
computational-stop rule therefore fired.

- folds required: 12
- folds converged: 0
- fold iterations: `candle=5000`
- complete R1 predictions produced: no
- R1 scientific metrics produced: no
- original R1 scientific gate evaluated: no
- R1 Recovery v1 decision: `COMPUTATIONAL_STOP`

The machine-readable provenance and terminal evidence are:

- `results/sabra_car/r1_recovery_v1/ATTEMPT_STARTED.json`
- `results/sabra_car/r1_recovery_v1/COMPUTATIONAL_STOP.json`

## Validation and firewall

- cheap canonical/R0/R1 suite: 21 passed
- frozen-fitter direct recovery contract assertions: PASS
- Phase2B training steps: 0
- MVTec accessed: no
- Medical accessed: no

Per Recovery Protocol v1, no second recovery, parameter change, incomplete R1
evaluation, R2-R4 execution, MVTec access, or Medical access is authorized.
