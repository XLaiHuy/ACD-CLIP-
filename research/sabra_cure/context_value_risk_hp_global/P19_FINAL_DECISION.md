# P19 Final Decision

`P19_ENGINEERING_STOP`

One P19 marker was created. `candle` completed its outer-fold checkpoint;
`capsules` completed 11 source-target packages then wrote `FAILED` before an
outer-fold checkpoint. The parent stopped exactly as required, with no audit,
global metric, final aggregation, P14 gate evaluation, or scientific
interpretation.

The child peak RSS values were 6,459,793,408 bytes (`candle`) and
6,604,632,064 bytes (`capsules`), both below the 14 GiB limit. The exact
engineering cause is recoverable without a rerun: `capsules` selected
`NO_EXPANSION`, so the controller represented its value threshold as
`float('inf')`; the atomic JSON writer has `allow_nan=False` and failed while
serializing that non-finite value. The forensic tempfile is truncated exactly
at `value_threshold`. The 4.6 GiB `capsules` map/target directory and its
partial JSON are fold-local temporary caches, not historical evidence; they
are excluded and moved to recoverable trash after this terminal record. The
complete `candle` checkpoint and both worker status files are retained as
provenance only.

MVTec, Medical, new CLIP forwards, and Phase2B optimization remain zero.
