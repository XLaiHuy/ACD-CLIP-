# Recovery Implementation Contract

The runner must reject an existing recovery marker, recovery summary, or
recovery terminal decision.  Before the marker, schema, hash, class/image/patch
alignment, finite input, source-contract hash, `.25` alpha, and P12 immutability
are audited.  After the marker, any uncaught exception atomically writes
`ENGINEERING_FAILURE.json` with exception type/message, stage, last class,
execution SHA, marker state, and a traceback log, then exits nonzero.

The runner records `PROGRESS.json` after each completed held class.  Progress
and logging are operational evidence only and cannot influence diagnostic
arrays or classification.  Final result and post-audit serialization is atomic.

Required tests cover provenance/contract protection, P12 immutability, the
prior missing durable lifecycle, D0--D4 masks, signs, bins/searchsorted,
synthetic AP/ranking/spatial/class aggregation, finite and deterministic
serialization, atomic failure capture, fixed alpha/no threshold, firewall, and
exactly-once recovery protection.
