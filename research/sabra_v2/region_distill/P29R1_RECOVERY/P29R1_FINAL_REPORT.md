# P29R1 FAST FORENSIC FINAL REPORT

## IDENTITY

- P29 terminal SHA: `7eeee454538cb997496f8cd1107f66fa73a9c876`
- P29R1 prereg SHA: `9cd75210fa288c5f95816bd275844555688a3c55`
- P29R1 execution-base SHA: `43f5792b1fb779acfdaf9a0ab73ddbc42e6a34e2`
- Forensic UUID: `b9f4a9e6-6d3a-4d10-9c2a-4b2a69c5e8f1`

## PERFORMANCE

- Runtime: `197.888` seconds for `12/12` classes.
- Peak RSS: `6908100` KiB; peak GPU allocated: `3047198208` bytes.
- New CLIP forwards: `0`; new Phase2B forwards: `0`; training steps: `0`; optimizer steps: `0`.

## SIGN / MAGNITUDE

- Frozen P27 sign agreement: `0.5228332297`; P29 sign agreement: `0.5195621046004545`; delta: `-0.0032711250995455243`.
- P29/P27 mean residual-magnitude ratio: `0.8723438446565194`.

## ZERO-INIT GRADIENTS

- L_value: `0.039433477948271435`; L_sign: `0.0`; L_normal: `0.0`.
- P27 raw distillation: `0.18329280901365844`; P29/P27 value ratio: `0.2876094472747026`.
- Zero-init classification: `{'g_normal': 'ZERO', 'g_sign': 'ZERO', 'g_value': 'NONZERO'}`.

## NORMALITY / RECOVERY

- Pure-normal teacher-positive fraction: `0.06292696755068998`; positive strength mass: `0.06275717619185646`.
- P27/P29 normal q99 shift: `0.0007015705637864011` / `0.0001596801139287361`.
- P27/P29 OR pAP recovery: `-0.972539238164763` / `-0.7460964997038254`.
- P27/P29 OR pAUROC recovery: `-0.458351859997298` / `-0.46326509826904205`.

## ROOT CAUSE DECISION

- `GRADIENT_STARVATION`: `SUPPORTED`
- `MIXED_OBJECTIVE_CONFLICT`: `SUPPORTED`
- `NORMALITY_FIX_INSUFFICIENT`: `SUPPORTED`
- `NORMAL_GUARD_CONFLICT`: `SUPPORTED`
- `REGION_REPRESENTATION_LIMIT`: `NOT_SUPPORTED`
- `SEGMENTATION_ANCHOR_REMOVAL`: `NOT_SUPPORTED`
- `SIGN_FIX_INSUFFICIENT`: `NOT_SUPPORTED`
- `STUDENT_CAPACITY_LIMIT`: `NOT_SUPPORTED`

Primary root cause: `MIXED_OBJECTIVE_CONFLICT`.
Secondary root cause: `GRADIENT_STARVATION`.
One recommended next research direction: `SINGLE_OBJECTIVE_DIRECTIONAL_DISTILLATION`.

All reported quantities use frozen P27/P29 artifacts, the P28R1-compatible held teacher definition, and bounded source-only gradient probes. No model was trained or updated.

## STATUS

`P29R1_FORENSIC_COMPLETE`
