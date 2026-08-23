# P18 Final Decision

`P18_AGGREGATION_NO_GO`

The pre-preregistration aggregation-sufficiency audit found that the frozen
P14 global stable-rank Spearman metric requires cross-fold ordering of every
image-level `V_j` and value prediction.  A compact scalar fold summary cannot
reconstruct it exactly, while the P18 parent is explicitly forbidden from
reading `V_j` or prediction arrays.  No compliant workaround exists without
changing the supplied P18 ownership contract or a frozen P14 reported metric.

Accordingly, no P18 preregistration, implementation, execution-base commit,
attempt marker, scientific fold worker, audit worker, scientific metric, or
P14 gate evaluation was created.  P17 remains immutable and its partial
`capsules` target remains forensic-only.
