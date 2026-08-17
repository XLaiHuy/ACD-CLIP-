# P5FR1CE1 evaluator recovery defect review

## Scope and provenance

This review is a pre-GT recovery artifact for branch
`autopilot/p5-fr1c-e1-evaluator-recovery`, created from the remotely verified
GT-free commit `cd0d22072d34804494e38fa95bc2e9d338bead7c`.

The historical P5FR1C invocation remains terminally invalid as an evaluator
implementation event: `P5FR1C_EVALUATOR_INVALID`. That invocation failed
before GT or mask access, before scientific metrics, and with zero model
forwards. This recovery does not rewrite that history and does not change the
frozen scientific protocol.

This review inspects only committed evaluator source and pre-GT structural
behavior. It does not inspect GT metrics or scientific result artifacts.

## `load_configs()` contract

`load_configs()` returns exactly:

```text
families, rows, ids
```

where:

```text
rows = [(family, cfg), ...]
```

with 26 unique config IDs in frozen family order:

```text
PCRR (8), CSRC (8), ASR (6), PGM (4)
```

The function validates `len(rows) == 26` and uniqueness, and returns the
already reconstructed ordered `ids` list as its third value.

## Confirmed defect

The failed evaluator used this expression in `integrity_subchecks()`:

```python
expected_ids = [x["config_id"] for x in load_configs()[1]]
```

`load_configs()[1]` is a list of `(family, cfg)` tuples, so tuple indexing by
`"config_id"` raises:

```text
TypeError: tuple indices must be integers or slices, not str
```

The minimal structural correction is:

```python
expected_ids = [cfg["config_id"] for _, cfg in load_configs()[1]]
```

or equivalently reusing the third return value. The recovery implementation
uses the former form to make the row contract explicit while preserving exact
frozen ordering.

## Complete structural review

All `load_configs()` call sites were inspected:

| Location | Use | Finding |
|---|---|---|
| `integrity_subchecks()` | Reconstruct expected IDs from rows | Confirmed tuple/dict defect above; pre-GT fix only |
| `evaluate()` config grouping | `for fam, c in config_rows` | Correct tuple destructuring |
| `evaluate()` index map | `for i, (_, c) in enumerate(config_rows)` | Correct tuple destructuring |
| `evaluate()` fold class rows | Reads config dictionaries from family lists | Correct |

The remaining evaluator was reviewed for structurally similar issues:

- `load_configs()` validates the exact 26 unique IDs and preserves family
  ordering.
- `load_folds()` checks the exact frozen five-fold class assignment.
- Evidence is required to have shape `[26, 1369]` and finite values before
  downstream metric code.
- Patch-grid upsampling uses the frozen `37 x 37` grid and `518 x 518`
  image size.
- GT mask loading is isolated in `load_mask()`, requires a labeled row and a
  `ground_truth` path, and is not reachable from `integrity_subchecks()`.
- The pre-GT integrity gate reads only `INPUT_LOCK.json` and
  `GT_FREE_DERIVED_MANIFEST.json`; it does not open images, masks, or GT
  metric files.
- `main()` requires the explicit `--allow-gt` flag, initializes the run
  status before evaluation, and records failure without fabricating a PASS.
- Scientific selection, bootstrap seeds, sign-flip semantics, Holm
  correction, gates, winner rules, and research-value rules were not changed
  or identified as structural defects.
- No checkpoint loader, model construction, `model.forward`, training path,
  medical path, evidence derivation, or B1 regeneration is present in the
  evaluator startup/integrity path.

No additional defect requires a scientific-rule or protocol change. The
recovery scope therefore remains limited to the tuple-indexing implementation
defect and its pre-GT tests.

## Recovery test plan

`tools/test_p5fr1ce1_evaluator_recovery.py` covers:

- **T01:** `load_configs()` returns exactly 26 unique IDs with frozen family
  counts and ordering.
- **T02:** IDs reconstructed from `(family, cfg)` rows exactly equal the
  third `load_configs()` return value.
- **T03:** `integrity_subchecks(config_ids)` completes without `TypeError`.
- **T04:** The integrity gate's reconstructed config ordering is exact.
- **T05:** Evaluator startup reaches the pre-GT integrity gate and stops at a
  synthetic pre-GT sentinel before class/metric evaluation.
- **T06:** Mask/image-open guards remain unused throughout recovery tests.
- **T07:** The frozen derivation/manifest counters remain at zero model
  forwards, with the pre-GT gate unable to invoke the deployment path.

The baseline suite is expected to expose T03–T05's known defect before the
implementation commit. After the minimal fix, all T01–T07 must pass without
opening GT, masks, or computing evaluator metrics.

## Freeze boundary

This review and the tests must be committed before the repaired evaluator is
committed. After the repaired evaluator is pushed and the single recovery GT
evaluation begins, no evaluator/source/statistical changes are permitted.
