# SABRA-CAR Final Decision

Status: STOP AT R1

Timestamp: 2026-08-23T02:46:12+07:00

## Decision

SABRA-CAR stops at Stage R1. R0 passed exactly, but the first full
leave-one-class-out multinomial action-predictor fold exhausted the frozen
`max_iter=1000` LBFGS budget and failed the preregistered correctness gate
before any held-out prediction was written.

The model family, solver, preprocessing, and iteration limit were frozen before
R1 fitting. No alternate solver, larger iteration budget, resampling, neural
predictor, or post-hoc fallback is authorized. Therefore R1 OOF evaluation,
R2, R3, R4, freeze, MVTec development evaluation, and retrospective Medical
evaluation are all `NOT_RUN`.

## Evidence

- R0 decision commit: `571c26b07209c3005837deebfc306f1e3d1b433b`
- R0 decision: CONTINUE
- R0 selected alpha: 0.25
- R0 signed minus matched positive-only pAP: +1.0319407402895497 pp
- R0 signed-better breadth: 12/12
- R0 signed minus matched positive-only pAUROC: +0.23619865972595022 pp
- R1 contract commit: `da8ac2b5cf137f96ac8ddda1c11739d9c0f13ad2`
- R1 implementation commit: `08fbc7bd6334e6f7e9ddfef56ba00fbe2cb37d3a`
- R1 manifest-fix commit: `389246d610c78fa5e679f11e980412349be51345`
- R1 fit-authorization commit: `58312d4`
- failing fold: held-out `candle`
- observed estimator classes: `[-1,0,1]`
- observed iterations: 1,000
- failure: LBFGS reached the total iteration limit
- fold predictions written: none
- thresholds evaluated: none
- source masks opened during R1: no
- Medical accessed: no
- MVTec accessed: no
- Phase2B training steps: 0

## Scientific interpretation

R0 establishes that oracle signed suppression carries incremental pixel-ranking
value, but this preregistered GT-free multinomial predictor was not numerically
certified under its frozen training contract. The workflow therefore makes no
claim about deployable CAR efficacy and does not advance to reliability,
radius, development, freeze, or final benchmarks.

A future attempt requires a new preregistration before observing new results;
it must not mutate this stopped run or reuse Medical outcomes for selection.
