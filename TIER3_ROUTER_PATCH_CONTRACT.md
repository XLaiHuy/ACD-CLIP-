# Tier-3 router patch contract

The dense router consumes the segmentation-token stream from the Phase-2B
visual forward, not `seg_tokens_pre_l2`, detector tokens, router queries, or
any pooled feature:

```text
visual_output["seg_tokens"]                         [G, B, P, 768]
PatchRouter.router_input_features(...)
  = F.normalize(tokens.float(), dim=-1)              [G, B, P, 768]
PatchRouter._local_query_inputs(...) -> query
softmax(query @ final_router_keys.T / temperature)   [G, B, P, M]
```

`router_input_features` is returned as
`h6_batch["router_patch_features"]`.  The patch-bank builder saves flattened
rows of this exact FP32, L2-normalized tensor from the real `train` dataset.
Tier-3 targets are detached soft assignments of those rows to the four saved
centroids; the KL compares them only with `dense_probabilities`, never the
scheduled or sparse prediction view.

Centroid index `m` is preserved when binding identity: projected centroid `m`
initializes `concept_slots[m]` (therefore semantic queries and
`router_key(concept_slots[m])`) and `factor_id_embedding[m]`.  Tier-3 requires
the factor-generator specialization flag so all three paths exist.
