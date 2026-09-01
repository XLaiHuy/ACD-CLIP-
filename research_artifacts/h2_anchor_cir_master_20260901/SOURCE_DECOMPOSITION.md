# H2 Master Source Decomposition

Status: PASS (source-only audit).

The gate uses the frozen deterministic VisA sample only. All arms use the historical H2 native deployment with alpha=0; RCA CIR is train-time only. Therefore this file does not estimate an inference-time RMT effect.

| method | epoch | pixel AUROC | pixel AP | image AUROC | image AP |
|---|---:|---:|---:|---:|---:|
| R | 10 | 0.94393531 | 0.40508183 | 0.94270833 | 0.96317448 |
| RA | 10 | 0.97697824 | 0.49089039 | 0.92230903 | 0.94271903 |
| RCA | 10 | 0.93112223 | 0.49341194 | 0.95920139 | 0.96937034 |
| RA | 12 | 0.95990398 | 0.49319691 | 0.92578125 | 0.94470241 |
| RCA | 12 | 0.93345608 | 0.49347167 | 0.95963542 | 0.96981455 |
| RA | 14 | 0.97303536 | 0.49106342 | 0.92621528 | 0.94581688 |
| RCA | 14 | 0.92923814 | 0.49464477 | 0.95833333 | 0.96818254 |
| RA | 16 | 0.97764765 | 0.49103001 | 0.92621528 | 0.94518562 |
| RCA | 16 | 0.93110247 | 0.49483609 | 0.95833333 | 0.96749987 |
| RA | 18 | 0.96044460 | 0.49191661 | 0.92708333 | 0.94617071 |
| RCA | 18 | 0.92998407 | 0.49467996 | 0.95572917 | 0.96526331 |
| RA | 20 | 0.94778058 | 0.49262208 | 0.92578125 | 0.94473767 |
| RCA | 20 | 0.90594481 | 0.49458322 | 0.95572917 | 0.96494194 |

R is the native H2 control. RA isolates the image-parameter anchor training effect relative to R. RCA adds train-time CIR relative to RA.

Anchor-gradient ratio: NOT_MEASURED. The trainer records total loss components and RCA peer/delta telemetry, but not separate per-objective gradient norms.

Medical and MVTec: NOT_RUN. Candidate continuation decisions are target-blind and must be recorded before target evaluation.

See SOURCE_DECOMPOSITION.csv, SOURCE_DECOMPOSITION_PER_CLASS.csv, SOURCE_PARAMETER_DRIFT.csv, and SOURCE_FEATURE_DRIFT.csv for the compact evidence tables.
