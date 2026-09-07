# H2 Root-Cause / S2-LOCR R1 Feature and Logit Provenance

The graph below is reconstructed from the frozen implementation before any diagnostic interpretation.

```text
CLIP patch tokens (37x37, 1024)
  -> Conv-LoRA raw output -> norm-matched AddWeight output
  -> seg_proj (1024->768) -> LayerNorm -> L2 normalized seg tokens
  -> DFG GAP/Q/K (+ SS2D weight residual)
  -> stage normal/abnormal logits (37x37)
  -> Gaussian7/sigma1 -> bilinear518 stage maps
  -> equal pre-softmax fusion -> final normal/abnormal logits/probability
```

| Node | Shape / resolution | Space | Source / semantics |
|---|---|---|---|
| `clip_patch_stage1` | `[B,1369,1024]` / `37x37` | `feature; projected margin for audit` | `model/adapter.py:238-270; ViT transformer output after selected frozen resblock and before Stage-1 trainable image adaptation |
| `clip_patch_stage2` | `[B,1369,1024]` / `37x37` | `feature; projected margin for audit` | `model/adapter.py:238-270; selected resblock output before Stage-2 adapter; stream already includes prior stage replacement |
| `clip_patch_stage3` | `[B,1369,1024]` / `37x37` | `feature; projected margin for audit` | `model/adapter.py:238-270; selected resblock output before Stage-3 adapter; stream already includes prior stage replacements |
| `convlora_raw` | `[B,1369,1024]` / `37x37` | `feature; projected margin for audit` | `model/adapter.py:270-291; model/adapter_modules.py:79-92; learned Conv-LoRA output before norm matching and AddWeight |
| `post_convlora` | `[B,1369,1024]` / `37x37` | `feature; projected margin for audit` | `model/adapter.py:270-291; normalized learned adapter merged with base through m_i_w |
| `pre_seg_projection` | `[B,1369,1024]` / `37x37` | `feature; projected margin for audit` | `model/adapter.py:293-300; group token representation entering seg_proj |
| `post_seg_projection` | `[B,1369,768]` / `37x37` | `normalized feature; direct text margin` | `model/adapter.py:300-309; MLP projection, LayerNorm, then L2 normalization |
| `dfg_input` | `[B,1369,768]` / `37x37` | `normalized feature; direct text margin` | `model/adapter.py:535-555; same normalized segmentation tokens entering DFG |
| `dfg_qk_compatibility` | `[B] broadcast to 37x37` / `global/broadcast` | `abnormal-minus-normal Q/K compatibility` | `model/adapter.py:555-607; global DFG query/key compatibility before weighted text-value fusion |
| `ss2d_input` | `[B,768] broadcast to 37x37` / `global/broadcast` | `direct text margin` | `model/adapter.py:544-552; GAP visual vector entering the SS2D residual path |
| `ss2d_output` | `[B,768] broadcast to 37x37` / `global/broadcast` | `direct text margin` | `model/adapter_modules.py:166-188; SS2D branch output before its DFG weight residual |
| `stage_logits_pre_blur` | `[B,2,37,37]` / `37x37` | `normal/abnormal logit margin` | `model/adapter.py:365-369; per-stage production logits before Gaussian blur |
| `stage_logits_post_blur` | `[B,2,37,37]` / `37x37` | `normal/abnormal logit margin` | `model/adapter.py:372-379; per-stage logits after existing Gaussian7 sigma1 |
| `stage_map_post_resize` | `[B,2,518,518]` / `518x518` | `normal/abnormal logit margin` | `model/adapter.py:372-379; per-stage logits after existing bilinear resize align_corners=True |
| `final_fusion_logits` | `[B,2,518,518]` / `518x518` | `equal pre-softmax normal/abnormal margin` | `h2_clean/stage_fusion.py; model/adapter.py:382-385; equal mean of post-resize stage logits; evaluator emits sigmoid margin |

The feature-point audit uses the existing text embeddings and a production-compatible direct normal/anomaly margin after the stage segmentation projection. Q/K compatibility and SS2D global vectors are reported in their native diagnostic semantics and broadcast only for region accounting; they are not treated as spatial pixel logits.

Exact footprint rule: each native token `(r,c)` covers input rows `[14r,14r+13]` and columns `[14c,14c+13]`.
