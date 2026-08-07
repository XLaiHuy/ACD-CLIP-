# Iteration C Calibration Report (Windowed Accumulation = 6)

**Dataset**: VisA  
**Images**: 120 (20 windows of 6 microbatches)  
**Config Hash**: `b0c959c5b79ef3574d91e6075f03700bf55ed0872f68d464c6e06a3e39b75319`  
**Decision**: `READY_FOR_ITERATION_D`  

## 1. Summary Statistics (Window-Level)

| Metric | Task Loss | Route Loss | Factor Role Loss | Actual Local Loss |
|---|---|---|---|---|
| **Mean** | 2.1771 | 1.3857 | 0.147537 | 0.147539 |
| **P05** | 2.1631 | 1.3762 | 0.096983 | 0.096986 |
| **P95** | 2.1849 | 1.3940 | 0.211631 | 0.211619 |

## 2. Calibrated Lambda Coefficients

- `lambda_route` (1.5% target): `0.023564`
- `lambda_factor_role` (2.0% target): `0.283605`
- `lambda_actual_local` (1.5% target): `0.212705`

### Split-Half Window Stability Analysis (First 10 vs Second 10 Windows)
- Route Lambda H1/H2: `0.023601` / `0.023526` (diff: `0.32%`)
- Factor Lambda H1/H2: `0.293051` / `0.274737` (diff: `6.46%`)
- Actual Lambda H1/H2: `0.219791` / `0.206052` (diff: `6.46%`)
- **Stability Gate (<=20% diff)**: `PASSED`

## 3. Weighted Auxiliary Share Distribution

- Mean Total Auxiliary Share: `5.00%` (target 5.0%, operational range 4.0%–6.0%)
- P95 Total Auxiliary Share: `6.36%` (operational max <= 8.5%)

## 4. Role Support Breakdown

- Role 0 (Normal): `62974` patches
- Role 1 (Outside Anomaly): `100108` patches
- Role 2 (Core Anomaly): `948` patches
- Role 3 (Boundary Anomaly): `250` patches
- **Role Support Gate**: `PASSED`

## 5. Decision & Approved State

Final Decision: `READY_FOR_ITERATION_D`
