# Phase 3 - source-only rho gate

Status: `LOCKED_RHO_010`

The S1 rho screen compared anchor family budgets `rho=0.05`, `0.075`,
`0.10`, and `0.15` from fresh metadata-only forks of the authoritative
shared E1 checkpoint. Alpha 0.20, the fixed anchor lambda, CIR off, seed 0,
the five-epoch 50-batch screen, and locked TTA-F were held constant.

The selector used the VisA source category slice `candle`, `macaroni1`,
`pcb3`, and `pipe_fryum`. The primary criterion was source Pixel AP. As with
the alpha gate, candidates were trained on the full VisA training split, so
the source category slice is a diagnostic rather than a fully independent
held-out-training estimate.

| rho | source Pixel AP | source Pixel AUROC | source Image AP | source Image AUROC | finite | family partition |
|---:|---:|---:|---:|---:|:---:|:---:|
| 0.05 | 20.9387610 | 94.5603505 | 91.4982006 | 89.3412590 | PASS | PASS |
| 0.075 | 22.5753810 | 95.0972759 | 91.9409126 | 89.6036610 | PASS | PASS |
| 0.10 | **24.6335168** | **95.6550694** | 90.6629398 | 88.0986422 | PASS | PASS |
| 0.15 | 21.4368295 | 93.3873148 | **92.6252007** | **90.7510146** | PASS | PASS |

`rho=0.10` is locked for full E15 confirmation. It has the highest source
TTA-F Pixel AP, leading the next candidate rho 0.075 by 2.0581358 percentage
points. The final family telemetry reports complete partition coverage and
an effective active-family ratio of 0.0999997, consistent with the locked
budget. No Medical, MVTec, target labels, or target metrics were used.

Candidate checkpoint hashes, raw JSON hashes, telemetry, and fixed settings
are in [`RHO_SOURCE_GATE.json`](./RHO_SOURCE_GATE.json). The raw screen is
under `/tmp/r3_anchor_rho_screen_20260909`.

Stage S2 is full E15 confirmation of the locked alpha/rho candidate before
any target evaluation.
