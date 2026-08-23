# SABRA-CAR R1-v2 Final Decision

Status: SCIENTIFIC STOP — NO RISK-QUALIFIED THRESHOLD

## Decision

The single execution authorized by `SABRA-CAR R1 Scalable Solver Protocol v2`
completed all 12 frozen VisA LOCO fits with `solver=newton-cholesky` and
`max_iter=100`. Every fold converged in 14 iterations, produced finite
parameters and probabilities, and passed probability-normalization checks.

Complete LOCO predictions were therefore assembled and the original frozen R1
gate was evaluated. No threshold in `{0.50, 0.60, 0.70, 0.80, 0.90}`
satisfied the frozen coverage and opposite-sign risk requirements. The result
is `R1_V2_EXECUTION_STATUS=SCIENTIFIC_STOP` and
`R1_DECISION=STOP`.

The best-coverage filtered candidate, threshold 0.50, had coverage
0.24766384505864966, opposite-sign rate 0.14312238244000927, and relative
risk reduction -0.2587917829399904 versus the unfiltered opposite-sign rate
0.11369821790998477. Because no risk threshold qualified, the preregistered
procedure stopped before CAR intervention efficacy evaluation; pAP, pAUROC,
and breadth values are therefore `NOT_RUN`, not zero.

## Preserved history

- Original S0: `08ca99ff69d6d85184f5d145830876befb413628`
- R1-v1 terminal commit: `782b8b81aa5b03c88a4417e5d7106e19ceff83ce`
- R1-v2 preregistration: `fefeab35b58d4aa6be4ceddfaaa0994fa456d180`
- R1-v2 implementation: `435b778032c1fc246cecac63af4d1e597411eb4c`
- Original R1 remains `INCONCLUSIVE_SOLVER_FAILURE_BEFORE_SCIENTIFIC_RESULT`.
- R1-v1 remains `COMPUTATIONAL_STOP`.
- R1-v2 is the first completed scientific R1 gate result.

## Firewall and continuation

R2, R3, R4, MVTec, and Medical were not run or accessed. Phase2B training
steps remain zero. No downstream continuation is authorized by this STOP.
