# P27R1 Final Scientific Report

## Identity

- Attempt UUID: `884f7327-1135-491b-8c12-dc188455be2c`
- Scientific execution base: `de41b380449dcbc0b124f71f4f8fbb789e1a96f0`
- Frozen original P27 parent: `1151373f2c4968268f52cdc3e538c7ebcef7b8f0`
- Protocol: 12-class VisA leave-one-class-out, with only
  `RegionResidualAdapter` trainable.
- Durable scientific state: 12/12 training folds, 12/12 immutable
  predictions, 12/12 scored folds.

## OBSERVED

- Native macro pAP: `0.4525216034`.
- P27 macro pAP: `0.4613875663`.
- Delta macro pAP: `+0.0088659629`.
- Native macro pAUROC: `0.9345650496`.
- P27 macro pAUROC: `0.9203411939`.
- Delta macro pAUROC: `-0.0142238557`.
- pAP improved in 8/12 classes and regressed in 4/12; non-regressing count
  is 8/12.
- Median pAP delta: `+0.0011469157`.
- Best pAP delta: `+0.0764828841` (`cashew`).
- Worst pAP delta: `-0.0441689632` (`macaroni1`).
- The top positive class accounts for `39.7385%` of positive pAP gain; the
  top two account for `69.0791%`.
- pAUROC improved in 1/12 classes (`pipe_fryum`) and regressed in 11/12.
- All 12 predictions were frozen before scoring; all scoring artifacts report
  zero fit/teacher steps.
- Held GT reads before scoring: `0`; held mask reads before scoring: `0`.
- MVTec reads: `0`; Medical reads: `0`.

## INTERPRETATION

Source-only SABRA region correction distillation transferred to several held
VisA categories and produced a positive macro pAP change, but the typical
category gain was small, the worst categories regressed materially, and gains
were concentrated in cashew and fryum. The pAUROC guardrail declined
substantially in aggregate and in 11/12 categories. Therefore the result is
mixed rather than evidence of a broad, robust improvement over the frozen P26
native detector.

Final scientific classification: `P27_MIXED`.

No result-driven rerun, tuning, or post-outcome protocol change was performed.
