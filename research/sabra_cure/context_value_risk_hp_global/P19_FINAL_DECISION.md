# P19 Final Decision

`P19_ENGINEERING_STOP`

One P19 marker was created. `candle` completed its outer-fold checkpoint;
`capsules` completed 11 source-target packages then wrote `FAILED` before an
outer-fold checkpoint. The parent stopped exactly as required, with no audit,
global metric, final aggregation, P14 gate evaluation, or scientific
interpretation.

The child peak RSS values were 6,459,793,408 bytes (`candle`) and
6,604,632,064 bytes (`capsules`), both below the 14 GiB limit. The current
controller did not persist child stderr, so no exact Python exception can be
claimed without a prohibited rerun. The 4.6 GiB `capsules` map/target directory
is fold-local temporary cache, not historical evidence; it is excluded and
removed after this terminal record. The complete `candle` checkpoint and both
worker status files are retained as provenance only.

MVTec, Medical, new CLIP forwards, and Phase2B optimization remain zero.
