# CIR_DFG_RMT_V2 source confirmation

Status: V2_SIGN_CONFIRMED

This bounded confirmation uses VisA only, the fixed class-stratified 120-image subset, and the preregistered alpha grid 0/0.10/0.25/0.50. The V2 direction is abnormal - alpha*delta; normal + alpha*delta. Alpha remains PROVISIONAL and release lock remains FALSE.

Confirmation execution identity: config SHA `b3b1494371a6aa88512c59d9b2e29519462ce71943af22e4420a68c898f0a8f8`, candidate freeze SHA `a5ebd919bd86d752fb549ba312e5888cc6c4eb7be6303cfe81bdf51989bf24b4`.
Final V2 identity after binding the PASS freeze document: config SHA `31827a7c6eaac0ffc7f906909f28b4eb208e5639c498d85b9fe80cbe284b1be0`, freeze SHA `f6de6ee8f1998f591c077efeff50fa9741a9f8bad34603ba145ec54ef961ba86`.

| alpha | pixel AUROC | pixel AP | image AUROC | image AP |
|---:|---:|---:|---:|---:|
| 0.00 | 0.430918 | 0.004577 | 0.503333 | 0.593155 |
| 0.10 | 0.454278 | 0.005489 | 0.506667 | 0.598975 |
| 0.25 | 0.486811 | 0.012185 | 0.496667 | 0.606554 |
| 0.50 | 0.523760 | 0.012816 | 0.530000 | 0.617897 |

Decision rule: two or more nonzero alphas must improve both pixel metrics; one must keep image AUROC/AP drops <=0.02

V1 terminal remains immutable; no release gate or full training was launched.
