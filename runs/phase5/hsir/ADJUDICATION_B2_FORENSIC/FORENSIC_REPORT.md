# Phase5-B2 Bridge Forensic Report

## Preservation

The immutable B2 evidence commit `1714178aa34012031c4720ca1c9e901ae61c08b7` was pushed and verified at the same remote SHA. The original `ADJUDICATION_B2/` directory was not modified, no model forward or dataset inference was run, and no partial metrics were used.

## Persisted class bridge values

| class | W_aligned | W_shifted | matched pairs | coverage |
|---|---:|---:|---:|---:|
| candle | 0.500000 | 0.500000 | 77 | 0.987179 |
| capsules | 0.500000 | 0.500000 | 124 | 1.000000 |
| cashew | 0.500000 | 0.500000 | 318 | 0.963636 |
| chewinggum | 0.500000 | 0.500000 | 146 | 0.993197 |
| fryum | 0.500000 | 0.500000 | 456 | 0.940206 |
| macaroni1 | 0.500000 | 0.500000 | 92 | 1.000000 |
| macaroni2 | 0.500000 | 0.500000 | 115 | 1.000000 |
| pcb1 | 0.500000 | 0.500000 | 450 | 0.971922 |
| pcb2 | 0.500000 | 0.500000 | 190 | 1.000000 |
| pcb3 | 0.500000 | 0.500000 | 116 | 0.991453 |
| pcb4 | 0.500000 | 0.500000 | 179 | 0.983516 |
| pipe_fryum | 0.500000 | 0.500000 | 522 | 0.925532 |

All 12 values are exactly 0.5. Pair-level E values were not persisted, so their distribution cannot be recovered without inference.

## Trace result

The shape/index path is consistent: patch evidence, risk, validity, bins, and occupancy use flattened row-major `[1369]` arrays; positive and negative indices are disjoint. The shift is applied before pair evaluation and is not aliased with aligned evidence. The GT firewall is intact.

The defect is at the final W call. `matched_win(evidence, pos, neg)` indexes one evidence array with both index vectors. In `process_class`, the aligned call supplies `bridge_e_pos` as that one array and supplies `arange(n)` for both `pos` and `neg`; `bridge_e_neg` is never used. The shifted call has the same defect.

Minimal reproduction: `e_pos=[0.9,0.8]`, `e_neg=[0.1,0.2]` should produce `W=1.0`, but the committed call `matched_win(e_pos, arange(2), arange(2))` produces `W=0.5` because it compares each positive value with itself.

## Decision

`B2_BRIDGE_IMPLEMENTATION_DEFECT_CONFIRMED`

The zero-width bootstrap CI is an artifact of the exact self-comparison defect, not a scientific conclusion about within-image conditioning. The smallest audit-only fix is to compare the separately persisted positive and negative evidence arrays directly; it has not been applied here.

## Next step

Obtain approval for one corrected, preregistration-preserving full VisA TEST rerun after applying only that two-array bridge-comparison fix.
