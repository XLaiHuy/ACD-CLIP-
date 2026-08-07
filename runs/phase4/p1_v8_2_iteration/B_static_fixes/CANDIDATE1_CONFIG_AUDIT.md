# Candidate 1 Configuration Audit

**Hash (SHA-256):** e541282a951cf812b98ed354487e36d1e6ab0a0ca1f8747ef9ab06ab93740526

## Protocol Resolution
This config was derived directly from the canonical source protocol `tools/phase4_gated_protocol.py`, which is the repository's ground-truth definition for structural training and medical evaluation:

- **dataset**: `VisA` (Source: `tools/phase4_gated_protocol.py` train command line 73)
- **img_size**: 518 (Source: line 74)
- **n_groups**: 3 (Source: line 81)
- **dfg_mode**: `attn` (Source: line 88)
- **dfg_attn_tau**: 8.0 (Source: line 90)
- **use_ss2d_dfg**: `true` (Source: line 91)
- **dfg_ss2d_fusion**: `weight_residual` (Source: line 93)
- **batch_size**: 1 (Source: line 76)
- **grad_accum_steps**: 6 (Source: line 78)
- **precision**: `bf16` (Source: line 80)
- **global_text_mode**: `hard_anchor` (Source: Default fallback overriding hybrid soft prompt for Iteration B candidate)
- **local_factor_mode**: `center_spread` (Source: Explicitly mandated for Option A-prime)
- **local_center_mix**: 0.05 (Source: Iteration B fixed formula for the first probe)
- **local_factor_spread**: 0.10 (Source: Iteration B fixed formula)
- **num_factors**: 4 (Source: mandated for Option A-prime roles)
- **dense_prediction**: `true` (Source: dense routing is mandated for Option A-prime)
- **h6_logit_temperature**: 1.0 (Source: Fallback default, NOT 0.07 to prevent scaling issues absent a specific protocol command)
- **Candidate-1 Objective Switches**: All disabled (`load_bias`, `balance`, `cluster`, `functional_diversity`, `router_teacher`, `center_losses`, `experts`).
- **seed**: 0 (Source: line 79)
