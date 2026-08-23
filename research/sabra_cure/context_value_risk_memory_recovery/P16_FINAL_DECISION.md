# P16 Final Decision

## Terminal status

`P16_ENGINEERING_STOP`

P16's one authorized attempt (`eb5ad96e-e4cb-4abb-bd7f-7fbc8461d6e8`)
reached the frozen M1 post-finalization RSS gate after the first completed outer
fold. The measured post-finalize RSS was `2,748,723,200` bytes, exceeding the
per-fold pre-RSS plus 1 GiB allowance. The execution therefore stopped before
the next fold and before any P14 gate evaluation.

The cache-lifetime instrumentation did show source cache release (`10,987,474,944`
to `3,090,804,736` bytes after source policy selection), but that observation
does not relax M1 or make the partial fold scientifically interpretable.

No P16 partial target, policy, prediction, metric, gate, or ranking result is
a scientific result. P15 partial evidence remains excluded. The P16 attempt is
consumed; no resume/rerun or code edit is permitted on this branch.

Firewall/freeze: MVTec reads `0`; Medical reads `0`; additional CLIP forwards
`0`; Phase2B optimizer steps `0`. P14/P15 frozen scientific contract remains
unchanged.

Next allowed action: explicit user review before any separately versioned
engineering recovery.
