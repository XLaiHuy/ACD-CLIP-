# Phase 1 — source-only TTA gate

Status: `LOCKED_TTA_F`

The source selector is a deterministic category-held-out VisA gate with
`candle`, `macaroni1`, `pcb3`, and `pipe_fryum` held out. The frozen A15
checkpoint was evaluated with identity, horizontal-flip, vertical-flip, and
horizontal-plus-vertical-flip inference. Segmentation maps were inverse
transformed before averaging; classification logits were averaged before the
binary score was formed.

| policy | transforms | source Pixel AP | source Pixel AUROC | source Image AP | source Image AUROC | latency multiplier |
|---|---:|---:|---:|---:|---:|---:|
| TTA-I | 1 | 49.2830627 | 96.8915368 | 94.9870393 | 94.4925725 | 1× |
| TTA-H | 2 | 50.0511555 | 97.0006167 | 95.5384105 | 95.1621279 | 2× |
| TTA-F | 4 | **51.6232194** | 96.8990332 | **96.0718200** | **95.7005441** | 4× |

TTA-F is locked by the preregistered source Pixel AP primary criterion. Its
Pixel AUROC is within 0.10 points of the best policy and no finite-output or
image-collapse guard failed. Medical and MVTec are evaluated only after this
lock; target results cannot change the TTA choice.

Full per-category source rows and provenance are in
[`TTA_SOURCE_GATE.json`](./TTA_SOURCE_GATE.json). The target evaluation is
recorded separately under the locked-policy artifact.
