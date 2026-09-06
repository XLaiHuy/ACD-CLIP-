# H2 functional feature anchor bounded R1 decision

`BOUNDED_TEST_STATUS=FAIL`

## Contract validity

- The existing R1 protocol, fixed lambda, and 16-batch no-update parity
  preflight were reused; no recalibration was performed.
- Control `A_SHORT_R1`: 500 attempted, 498 successful, 0 nonfinite-loss
  skips, 2 nonfinite-gradient skips. This fails the preregistered numerical
  rule of at most one isolated recoverable gradient skip.
- Candidate `A_FUNC_SHORT_R1`: 500 attempted, 500 successful, 0 loss skips,
  0 gradient skips.
- File order, labels, transformed image hashes, and mask hashes match for all
  500 attempts.
- The candidate functional gradient remained auxiliary by the fixed cap:
  median effective ratio 0.10, maximum 0.10, with 494 capped steps.

## Fixed source-only endpoint gates

The clean 96-image VisA subset was fixed as the first four normal and first
four anomalous manifest-order images in each category. It was evaluated only
after both endpoints completed and was not used to tune any value.

- Stage-2 cosine-to-E1 geometry improved by `+0.029286`.
- Stage-3 cosine-to-E1 geometry improved by `+0.035591`.
- Final source AUROC fell by `0.108011`; final source AP fell by `0.017515`.
- Positive coverage mean fell from `0.127369` to `0.089658` and median also
  fell; interior coverage mean fell from `0.156930` to `0.118421`.
- Stage-1 AP fell slightly (`-0.000469`), so the stage-1 health gate is not
  a positive rescue signal.

## Scientific interpretation

`DID_MECHANISM_FAIL=YES`

`DID_DIAGNOSIS_FAIL=NO`

The result is a mechanism failure, not a falsification of the bottleneck:
the intervention moves the intended stage-2/stage-3 features toward E1 but
also suppresses source anomaly coverage and ranking under the fixed R1
budget. The numerical control failure independently prevents a PASS. No
lambda, rho, seed, optimizer, prompt, or horizon adjustment is authorized.

`MECHANISM_SUPPORTED=NO`

`NEXT_FULL_E15_JUSTIFIED=NO`

The frozen Top-3 table is not reopened in this turn. Candidate #2 is not run
because the secondary conflict diagnosis is only `POSSIBLE`, and candidate #1
has not been exhausted by an authorized direct replacement. No target metric
was consulted.

## Evidence

- `audit/H2_FUNC_ANCHOR_R1_PROTOCOL.json`
- `audit/H2_FUNC_ANCHOR_R1_PREFLIGHT.json`
- `audit/H2_FUNC_ANCHOR_R1_CALIBRATION.json`
- `audit/H2_FUNC_ANCHOR_R1_SOURCE_EVAL.json`
- `/workspace/h2_functional_anchor_bounded_r1/A_SHORT_R1/summary.json`
- `/workspace/h2_functional_anchor_bounded_r1/A_FUNC_SHORT_R1/summary.json`

