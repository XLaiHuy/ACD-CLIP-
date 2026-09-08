# H2 CAB-LoRA R1 Graph Provenance

The production path is `model/adapter.py:ACDCLIP.forward`. The ViT stream
enters Stage 2 at `t=x[1:]` after transformer block 16, shape
`[1369,B,1024]`, before `image_adapter["lora_adapters"][1]`. This is the
pre-adapter tensor `z_pre`; it is captured with a forward pre-hook only for
instrumentation and is detached for CAB.

The raw Stage-2 Conv-LoRA output is norm-matched to `t` exactly as the
production path does. CAB's `z_post` uses that output and the Stage-2
`m_i_w[1]` merge coefficient, but the coefficient and pre tensor are
detached. Consequently the CAB graph contains only Stage-2 Conv-LoRA
parameters. It stops before Stage-2 projection, DFG Q/K, SS2D, Stage 3,
stage fusion, and interpolation.

Stage-2 logits are the unchanged `_vision_text_attention_fusion` output
converted to `[B,2,37,37]`; the production score path is Gaussian 7x7,
bilinear resize to 518 with `align_corners=True`, equal pre-softmax fusion,
then softmax. CAB never edits these logits.

The identity artifact records exact disabled parity for Stage-1/2/3 logits,
final fused logits, final anomaly map, and task loss. CAB is disabled by
default and has no production-forward branch.
