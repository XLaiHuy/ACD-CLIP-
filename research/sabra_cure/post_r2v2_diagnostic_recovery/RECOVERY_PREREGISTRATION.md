# SABRA-CURE Post-R2v2 Actionability Diagnostic Engineering Recovery V1

Status: `FROZEN_BEFORE_RECOVERY_EXECUTION`.

Parent P12 terminal SHA: `cad96634c820950d459f6a170b8f629abd2e8040`.
P12 preregistration SHA: `413f26d4849b8db42d64be5c562aa6067a36e61c`.
P12 execution base SHA: `05bb08aae3fd35788ed0a70875185e32fdb7483c`.

P12's scientific contract is copied by hash and remains authoritative: H1--H7,
D0--D4, all oracle labels, alpha `.25`, actions/signs, five-bin
`searchsorted(..., side='right')` policy, aggregate-only AP analyses, T0--T3,
and its decision rules are unchanged.  This recovery has exactly one new,
recovery-local attempt marker.  It is not a P12 rerun identity and not a new
scientific method.

The frozen root cause is `PATH_ENVIRONMENT`: P12 lacked a durable post-marker
execution lifecycle.  The minimal repair is a recovery-local runner that uses
the same P12 analytical functions but atomically records marker/progress/result
or `ENGINEERING_FAILURE.json`, writes traceback/log evidence, and is launched
through its module entrypoint.  No numerical computation is changed.

Only these paths may change after this freeze: the recovery-local research and
result directories, `tools/sabra_cure/post_r2v2_diagnostic_recovery.py`, and
`tests/test_sabra_cure_post_r2v2_diagnostic_recovery.py`.  Exactly one recovery
execution is allowed.  MVTec and Medical reads, CLIP forwards, Phase2B training,
model fitting, alpha/threshold/coverage changes, R2-v3/R3/R4, and modification
of P12 scientific artifacts are forbidden.
