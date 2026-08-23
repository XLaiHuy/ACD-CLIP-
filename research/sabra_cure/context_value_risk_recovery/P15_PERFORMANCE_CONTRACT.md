# P15 Performance Contract

The benchmark contains only the frozen real fixtures specified in the exact
parity contract; it must not complete an outer fold or emit P15 macro/gate
outcomes. Median speedup is reference runtime divided by optimized runtime.
P15 may execute only when parity passes, median speedup is at least `5x`, and
the benchmark-derived projected full run is operationally reasonable (preferred
at most eight hours). Worker count is fixed to `min(4, available CPU count)`
after parity checks and output image order is deterministic.

The optimized engine caches masks, paths, flattened labels, SAFE20/E40 score
maps, SAFE20 grouped counts, and baseline pAP for one class/context at a time.
It never retains all outer contexts merely for speed.
