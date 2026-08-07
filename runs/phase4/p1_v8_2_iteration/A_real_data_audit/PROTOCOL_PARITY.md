# Protocol Parity

## Base Configuration (progress1_v8_structural_smoke_seed0)
- **Model**: ViT-L-14-336
- **Architecture**: `h6_progress_version='P1-v8-minimal'`
- **Global Anchor**: `h6_global_text_mode='hard_anchor'`
- **Routing**: `h6_prediction_routing='dense'`
- **Router Dimension**: `h6_router_dim=128`
- **Bank Dimension**: `h6_bank_dim=256`
- **Factors (M)**: `h6_num_factors=4`
- **Experts**: Disabled (`h6_expert_enabled=False`)
- **Residual Effects**: Enabled (`lambda_h6_visual_residual=0.01`, `lambda_h6_consistency=0.01`, `lambda_h6_center=0.1`)
- **Query Mode**: `local_global_bypass` with `h6_router_query_global_weight=0.1`

This establishes the baseline target configuration for iteration testing.
