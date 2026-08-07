# P1-v8.2 Iteration Evidence Ledger

## Stage 0: Repository and Protocol Audit
- **Commit**: `96c5b9c6ad8ec2b3b2eaec11a5b0deab58d41b2c`
- **Branch**: `phase4-progress1-cops-dynamic-prompt`
- **Canonical Protocol**: P1-v8-minimal with hard_anchor, dense routing, 4 factors, no experts, 0.1 center weight.
- **Status**: PASS

## Stage A: Real Data/Augmentation Audit
- **Objective**: Build `tools/audit_p1_v8_2.py` to audit real data, check factor specializations and responsibility assignment.
- **Status**: PENDING
## Audit Result: Role Specialization (adapter_3.pth)
- **Patch Router Probabilities**: The dense router assigns roughly equal probabilities (~0.25) to all 4 factors for every patch regardless of role (Normal vs. Anomaly Core/Boundary/Outside).
- **Entropy**: Entropy is 1.3863 ($\approx \ln(4)$) across all roles.
- **Conclusion**: The factor collapse hypothesis is heavily confirmed. The local experts/factors are completely unspecialized at this stage. There is no spatial variation in factor routing, leading to identical predictions across factors and no localized anomaly specialization.

