# Speed and resource report

Status: COMPLETE. Timings below are taken from the completed arm telemetry and resumable evaluation cells; no new profiling run was launched.

## Wall clock

| workload | measured seconds | measured hours | basis |
|---|---:|---:|---|
| R E1-E10 | 6305.420 | 1.752 | sum of 10 training telemetry rows |
| RA E1-E20 | 12727.412 | 3.535 | sum of 20 training telemetry rows |
| RCA E1-E20 | 12823.043 | 3.562 | sum of 20 training telemetry rows |
| six-target Medical, 18 cells | 4262.158 | 1.184 | sum of completed cell elapsed times |
| one-winner MVTec, 15 cells | 308.703 | 0.086 | sum of completed cell elapsed times |

R, RA, and RCA training totals include the historical AMP behavior and exact matched scheduler. R was intentionally stopped at E10 under FAST_RIGOR; RA and RCA resumed through E20.

## Resource evidence

The admission preflight recorded an RTX 5060 Ti with 16,311 MiB VRAM, approximately 31 GiB host memory, and approximately 175 GB free disk. A later single health check during Medical showed 4,894 MiB VRAM in use and 100% GPU utilization, with no OOM or duplicate evaluator. Peak allocator/RSS instrumentation was not part of this run, so a formal peak value is **NOT MEASURED** rather than inferred.

## Recovery and saved compute

Recovery count: **3** technical recoveries, with no scientific-setting change:

1. RA was relaunched once after the first command omitted the required H2 E1 anchor argument; no training checkpoint was produced by the failed invocation.
2. The first strict full source-gate invocation stopped at the intentionally absent R E12 checkpoint; it was rerun with the preregistered FAST_RIGOR R-stop policy.
3. Medical completed all 18 cells, then resumed once to repair summary formatting for undefined image metrics; completed cells were not recomputed.

FAST_RIGOR avoided the R E11-E20 trajectory (approximately 1.75 hours at the observed R rate), avoided full per-epoch Medical evaluation, avoided alpha/inference sweeps, and used one final MVTec winner only. Raw pixel stores, memmaps, caches, spools, checkpoints, and dataset/model assets are excluded from the tracked archive.
