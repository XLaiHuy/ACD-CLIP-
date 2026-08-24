# P23 Surgical P22 Controller Interface Recovery

P23 is the one authorized governance exception from P22 terminal
`5d869debdb245627fb6c389349489e51af08ecd5`.  P22 stopped at 0/12 folds before
any scientific outcome because `ClassCache.n_images` does not exist.

The only permitted code change is in a new P23 runner: replace each
`ClassCache.n_images` access with the canonical count `len(cache.paths)` and
assert `len(paths) == len(native) == len(safe) == len(expand) == len(seed2)`
before coordinate initialization.  P15/P20/P21/P22 are immutable references;
P23 must not add an artificial P15 field.

All P21/P22 science and performance semantics remain frozen: action families,
thresholds, alpha, coordinates, two seeds, sweeps, epsilon/ties, gates, F0/F1,
P0/P1/P2, LOCO, exact AP, CUDA backend and runtime ceiling.  P23 adds only
interface audit plus pre-marker real-artifact smoke and stub-controller
rehearsal.  A fresh P23 attempt is permitted only after every pre-marker check
passes; a subsequent post-marker failure stops permanently with no P24.

Firewall remains MVTec 0, Medical 0, additional CLIP forwards 0, Phase2B 0,
and no R2-v3/R3/R4.
