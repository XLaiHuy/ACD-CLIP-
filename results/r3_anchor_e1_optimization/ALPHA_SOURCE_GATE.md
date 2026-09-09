# Phase 2 - source-only alpha gate

Status: `LOCKED_ALPHA_020`

The S1 alpha screen compared `hybrid_alpha_max=0.15` and `0.20` from fresh
metadata-only forks of the authoritative shared E1 checkpoint. Anchor safety
was enabled with the fixed lambda and family budget, CIR was disabled, and
both candidates used the same five-epoch, 50-batch-per-epoch screen.

The selector evaluated the preregistered VisA source slice consisting of
`candle`, `macaroni1`, `pcb3`, and `pipe_fryum` with locked `TTA-F`. The
primary criterion was source Pixel AP. The candidates were trained on the
full VisA training split, so this is a source-category diagnostic rather than
a fully independent held-out-training estimate; the limitation is recorded
explicitly.

| alpha | source Pixel AP | source Pixel AUROC | source Image AP | source Image AUROC | finite |
|---:|---:|---:|---:|---:|:---:|
| 0.15 | 23.9123699 | 95.4143148 | 90.2475998 | 86.7123544 | PASS |
| 0.20 | **24.6335168** | **95.6550694** | **90.6629398** | **88.0986422** | PASS |

`hybrid_alpha_max=0.20` is locked for the next source-only rho screen. It
leads alpha 0.15 by 0.7211469 percentage points of source Pixel AP. No
Medical, MVTec, target labels, or target metrics were used for this choice.

Candidate checkpoint hashes, raw JSON hashes, fixed settings, and the full
selection record are in [`ALPHA_SOURCE_GATE.json`](./ALPHA_SOURCE_GATE.json).
The raw screen remains under `/tmp/r3_alpha_screen_20260909_v2`.

Stage S2 is full E15 confirmation of the locked source candidate before any
target evaluation.
