# P15 — P14 Exact-Equivalent Computational Recovery V1

P15 is an engineering recovery of P14, not a new hypothesis. Its parent is
P14 terminal engineering-stop commit `09be5fb0daf80b3697634b738f7579c7041af690`.
The sole P15 scientific attempt may start only after the parity, performance,
publication, and pre-execution audit conditions in this directory pass.

The scientific target remains exactly

`V_j = pAP(SAFE20 with image j replaced by EXPAND40) - pAP(SAFE20)`.

SAFE20/EXPAND40 use P14's leakage-safe `.20/.40` risk quantiles; the 16 frozen
image features, nested LOCO, float64 ridge (lambda 1, training median/IQR,
centered solve, unregularized intercept), candidate quantiles
`(.50,.60,.70,.80,.90)`, alpha `.25`, source selection, comparators, metrics,
and P14 gates are unchanged.

One P15 attempt has a UUID, immutable execution-base/prereg/input hashes, and
atomic checkpoints after source-fold groups, image-target groups, and outer
folds. Mechanical restart is permitted only from a validated checkpoint with
unchanged identity, code SHA, prereg SHA, inputs, and scientific fields; every
restart is appended to `RESUME_LOG.jsonl`. A code edit after its marker causes
`P15_ENGINEERING_STOP`.

No MVTec or Medical reads, CLIP forwards, Phase2B steps, alpha/threshold sweeps,
or scientific full-run benchmarks are permitted. Stop after P15 terminal
evidence and require explicit user review.
