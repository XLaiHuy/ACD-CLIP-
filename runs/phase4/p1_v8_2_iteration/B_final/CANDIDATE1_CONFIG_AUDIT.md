# Candidate-1 Configuration Audit

**Date**: 2026-08-06  
**Status**: RESOLVED

## Config Path
`configs/phase4/p1_v8_2_candidate1.json`

## SHA-256 (canonicalized JSON)
`b0c959c5b79ef3574d91e6075f03700bf55ed0872f68d464c6e06a3e39b75319`

## Key Corrections from Previous Version

| Field | Old Value | New Value | Justification |
|-------|-----------|-----------|---------------|
| `rho_values` | `[0.0, 0.0, 0.0]` | `[0.05, 0.05, 0.05]` | Zero rho disconnects correction objective |
| `h6_logit_temperature` | `1.0` | `10.0` | Model default is 10.0; config must document actual value |
| `correction_max` | `20.0` | `1.0` | Capacity audit: 2×T×rho = 2×10.0×0.05 = ±1.0 |
| `schema_version` | `"1.0"` | `"1.1"` | Reflects breaking corrections |
| `source_protocol` | missing | `"phase4-progress1-cops-dynamic-prompt"` | Required for source identification |
| `source_dataset` | missing | `"multi-medical"` | Required for source identification |
| `source_split` | missing | `"train"` | Required for source identification |
| `role_target_version` | missing | `"v1-hard-onehot"` | Required for role semantics |
| `role_morphology_config` | missing | `{"boundary_threshold": 0.01, "core_threshold": 0.99}` | Required for build_semantic_roles |
| `correction_max_mode` | missing | `"resolved_from_capacity_audit"` | Documents resolution method |

## Correction Capacity Evidence

The H6 logit is defined as:
```
h6_logit = temperature × (cos_abnormal - cos_normal)
```
where `temperature = 10.0` (confirmed from `model/h6/model.py:97,830`).

Cosine similarity is bounded by [-1, +1], so:
```
h6_logit ∈ [-2 × temperature, +2 × temperature] = [-20, +20]
```

With `rho = 0.05`:
```
rho_scaled_correction ∈ [-2 × temperature × rho, +2 × temperature × rho]
                       = [-2 × 10.0 × 0.05, +2 × 10.0 × 0.05]
                       = [-1.0, +1.0]
```

**Therefore `correction_max = 1.0` is the only valid upper bound.**

The old value of `correction_max = 20.0` exceeded the theoretical capacity by 20×.

## Observed Values (One-Batch Dry Run)

From the source-exact dry run on VisA:
- Actual correction range: [-0.012, +0.007]
- Saturation rate (|c| ≥ 0.95×cap): 0.00
- Zero saturation observed — the correction_max=1.0 bound is not tight for fresh initialization

## Strict Validation Rules (Implemented)

The config loader enforces:
1. `rho_values` must all be > 0
2. `rho_values` length must equal `n_groups`
3. `rho_trainable` must be false
4. `experts_enabled`, `load_bias_enabled`, `balance_enabled`, `cluster_enabled`, `functional_diversity_enabled`, `router_teacher_enabled`, `center_losses_enabled` must all be false
5. `global_text_mode` must be `"hard_anchor"` (not hybrid)
6. All required fields must be present

## Final Resolved Config
See `candidate1_resolved.json` in this directory.
