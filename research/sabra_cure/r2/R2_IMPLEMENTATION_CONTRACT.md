# SABRA-CURE R2 Implementation Contract v1

Implementation is isolated to `tools/sabra_cure/r2.py` and
`tests/test_sabra_cure_r2.py`. It imports immutable R1 pure feature/ridge
helpers only; it must not edit historical R0/R1/P8 code or artifacts.

Required modes are `--pre-audit`, `--execute-once`, and `--audit-only`.
`--audit-only` reconstructs only persisted results and cannot refit a fold.
All writes use temporary files followed by atomic replace. `--execute-once`
rejects any existing attempt, result, or terminal decision marker.

Tests cover: fixed feature/class order; float64 finite shapes; full versus
sufficient-statistic ridge parity; exact inner/outer exclusions; conservative
quantile index; strict interval boundaries; deterministic selector tie-break;
abnormal-logit-only fixed-alpha correction; zero-correction native parity;
serialization/reload parity; historical protected hashes; firewall counters;
and exactly-once protection. No test or runtime path opens MVTec or Medical.
