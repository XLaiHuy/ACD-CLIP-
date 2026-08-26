# P29R1 Recovery Root-Cause Note

This note records the forensic finding for the terminalized original P29R1 attempt. It is a new recovery artifact; the original failure, audit, and final-report files are preserved unchanged.

## Exact failure

The original attempt stopped with:

```text
NameError: name 'residual_magnitude_summary' is not defined
```

The helper is defined in `tools/sabra_v2/p29r1_forensic.py`. The production entrypoint `tools/sabra_v2/run_p29r1_forensic.py` called it three times while constructing `_held_class_result`, but its import list omitted the helper. The first call was reached after the first held class's teacher and student region arrays had been computed, when the class-result dictionary was evaluated.

## Why existing checks passed

The prior 12 tests imported and exercised the helper directly and checked static contracts, but did not invoke the production `run()` path through `_held_class_result`. The preflight exercised source-only engineering probes and likewise did not execute the held-class result construction. Therefore neither check could expose the entrypoint-only missing import.

## Classification and fix

This is an entrypoint-only missing-import failure, not a circular-import, wrong-module, packaging, or frozen-input failure. The minimal scientific-path fix is to import `residual_magnitude_summary` in the runner. A separate engineering-path fix routes the final report through the requested output directory, so a recovery run cannot overwrite the original root report.

The exact regression test failed before the import fix with the same `NameError`; after the fix it reached a sentinel immediately after the held-class result, proving that the original failure path is repaired without running a full forensic experiment.
