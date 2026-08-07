# Role Specialization Audit

- **Checkpoint**: `runs/phase4/progress1_v8_structural_smoke_seed0/adapter_3.pth`
- **Dataset**: Brain (test)
- **Samples Evaluated**: 200

## Patch Counts by Role
- **Role 0 (Normal)**: 41070
- **Role 1 (Anomaly Outside)**: 218885
- **Role 2 (Anomaly Boundary)**: 8075
- **Role 3 (Anomaly Core)**: 5770

## Average Local Usage (Factor × Role)
| Factor | Role 0 (Norm) | Role 1 (Out) | Role 2 (Bound) | Role 3 (Core) |
|--------|---------------|--------------|----------------|---------------|
| Factor 0 | 0.2500 | 0.2500 | 0.2499 | 0.2499 |
| Factor 1 | 0.2500 | 0.2500 | 0.2500 | 0.2501 |
| Factor 2 | 0.2498 | 0.2499 | 0.2506 | 0.2508 |
| Factor 3 | 0.2502 | 0.2502 | 0.2494 | 0.2492 |

## Factor Entropy by Role
- **Role 0**: 1.3863
- **Role 1**: 1.3863
- **Role 2**: 1.3863
- **Role 3**: 1.3863
