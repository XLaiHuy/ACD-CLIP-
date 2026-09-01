# H2 Master E10 Source Gate

Status: PASS (source-only audit).

The gate uses the frozen deterministic VisA sample only. All arms use the historical H2 native deployment with alpha=0; RCA CIR is train-time only. Therefore this file does not estimate an inference-time RMT effect.

| method | pixel AUROC | pixel AP | image AUROC | image AP |
|---|---:|---:|---:|---:|
| R | 0.94393531 | 0.40508183 | 0.94270833 | 0.96317448 |
| RA | 0.97697824 | 0.49089039 | 0.92230903 | 0.94271903 |
| RCA | 0.93112223 | 0.49341194 | 0.95920139 | 0.96937034 |

R is the native H2 control. RA isolates the image-parameter anchor training effect relative to R. RCA adds train-time CIR relative to RA.

Anchor-gradient ratio: NOT_MEASURED. The trainer records total loss components and RCA peer/delta telemetry, but not separate per-objective gradient norms.

Medical and MVTec: NOT_RUN. Candidate continuation decisions are target-blind and must be recorded before target evaluation.

See E10_SOURCE_GATE.csv, E10_SOURCE_GATE_PER_CLASS.csv, E10_SOURCE_PARAMETER_DRIFT.csv, and E10_SOURCE_FEATURE_DRIFT.csv for the compact evidence tables.
