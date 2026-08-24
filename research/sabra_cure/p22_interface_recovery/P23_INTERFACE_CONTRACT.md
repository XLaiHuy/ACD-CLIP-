# P23 Interface Contract

Canonical `ClassCache` image count is `len(cache.paths)`.  Before every P23
coordinate invocation require equality of paths, native, safe, expand, and the
second-seed length.  `RecoveryEngine.n_images` remains valid because it is
constructed from `len(native)`.

The pre-marker audit enumerates every production attribute on ClassCache,
RecoveryEngine, fold packages, flat delta store and backend objects.  Every
P23 callsite must name an existing attribute with an expected shape/type.  The
real candle smoke calls the production `execute_once -> run_action_space ->
witness` construction path but intercepts before coordinates.  The separate
stub rehearsal executes every routing and serialization branch with no real
scientific metric.
