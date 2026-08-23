# P17 — Process-Isolated Exact P14 Recovery V1

P17 is a separately versioned engineering recovery from P16 terminal
`7eefaf69ae7c59761136b5ddf65fd82b4434ce2b`. It preserves exactly the frozen
P14/P15/P16 scientific contract: folds, target, features, float64 ridge,
lambda 1, q set, source selection, alpha `.25`, comparators, metrics, gates,
exact float32 AP/tie semantics, nested exclusions, and firewall.

P17 changes execution ownership only. Each held outer fold runs in one child
process; its address space is reclaimed on exit. The parent retains only attempt
identity, input hashes, completion metadata, compact scalar summaries, and
subprocess state. One P17 attempt is partitioned into 12 deterministic workers,
not 12 attempts. P15/P16 partial outputs are excluded from P17 science.
