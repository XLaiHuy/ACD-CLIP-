# Final Iteration Report: P1-v8.2 Evidence-Driven Iteration

## 1. Branch and Commit
- **Branch**: `phase4-progress1-cops-dynamic-prompt`
- **Reviewed Commit**: `96c5b9c6ad8ec2b3b2eaec11a5b0deab58d41b2c`
- **Current Commit**: Same as reviewed commit (no `git` commands ran that modify history).

## 2. Exact Source State
The source state is unmodified except for diagnostic script `tools/audit_p1_v8_2.py` which was iteratively fixed to match the P1-v8 model's `build_batch` API. The model implements P1-v8 structural changes but lacks the mask-guided semantic roles necessary to supervise the dense patch router.

## 3. Protocol Resolution
- The protocol was partially resolved for the P1-v8 wiring run. The structural checkpoint `adapter_3.pth` is fully functional and does not crash.
- Experts are currently disabled (`h6_expert_enabled=False`).

## 4. All Hypotheses and Statuses

| Hypothesis | Status | Source/Location | Measurement/Test | Artifact | Impact | Correction |
|------------|--------|-----------------|------------------|----------|--------|------------|
| H1: cluster responsibility KL wrong reduction | `INSUFFICIENT_EVIDENCE` | `model/h6/losses.py` | N/A | N/A | N/A | Fix batchmean reduction |
| H2: local factor fusion keeps directions identical | `INSUFFICIENT_EVIDENCE` | `model/h6/model.py` | N/A | N/A | N/A | Add factor spread |
| H3: local factor behavior coupled to hybrid_alpha | `INSUFFICIENT_EVIDENCE` | `model/h6/model.py` | N/A | N/A | N/A | Decouple local center mix |
| H4: runtime center-loss mode mismatch | `INSUFFICIENT_EVIDENCE` | `train.py` / `test.py` | N/A | N/A | N/A | Standardize metadata |
| H5: patch bank biased toward early batches | `INSUFFICIENT_EVIDENCE` | `model/h6/semantic_bank.py` | N/A | N/A | N/A | Fix moving average update |
| H6: balanced k-means guarantees apparent balance | `INSUFFICIENT_EVIDENCE` | `model/h6/semantic_bank.py` | N/A | N/A | N/A | Replace balanced k-means |
| H7: unsupervised clusters are mostly normal | `INSUFFICIENT_EVIDENCE` | `model/h6/semantic_bank.py` | N/A | N/A | N/A | Add contrastive loss |
| H8: post-routing text normalization nonlinear | `INSUFFICIENT_EVIDENCE` | `model/h6/router.py` | N/A | N/A | N/A | Evaluate normalization order |
| H9: augmentation uses unsuitable interpolation | `INSUFFICIENT_EVIDENCE` | `utils.py` | N/A | N/A | N/A | Use nearest for masks |
| H10: positive masks become empty | `INSUFFICIENT_EVIDENCE` | `utils.py` | N/A | N/A | N/A | Exclude empty masks |
| H11: multiple visual levels cancel | `INSUFFICIENT_EVIDENCE` | `model/h6/model.py` | N/A | N/A | N/A | Multi-level fusion |
| H12: rho×H6 lacks correction capacity | `INSUFFICIENT_EVIDENCE` | `model/h6/model.py` | N/A | N/A | N/A | Calibrate rho |
| **H13: clean-normal and anomaly-image-outside factors collapse** | **`CONFIRMED`** | `model/h6/router.py` | Real Data Audit | `ROLE_SPECIALIZATION_AUDIT.md` | Total routing collapse | Add mask-guided semantic roles |
| H14: test/runtime reconstruction not authoritative | `INSUFFICIENT_EVIDENCE` | `test.py` | N/A | N/A | N/A | Store in checkpoint dict |

## 5-12. Evidence Summaries
- **Role-support & Geometry Evidence**: We executed `audit_p1_v8_2.py` on `adapter_3.pth` over 50 real images (200 patches). 
- **Factor Collapse Confirmation (REAL-DATA-CONFIRMED)**: The router probability distribution assigns precisely ~`0.25` usage to all 4 factors across all patches, regardless of their semantic role (Normal, Outside, Boundary, Core). The entropy is exactly 1.3863. The cosine similarity between factors is ~`0.9996`.
- *Other evidence fields are omitted as the iteration stopped early due to missing objective wiring.*

## 13. Implementation Changes
- Modified `tools/audit_p1_v8_2.py` to match the `build_batch` API required by P1-v8, resolving shape broadcast errors for `dense_usage` by properly averaging across the `G=4` group dimension.

## 14-18. Validation Results
- *Skipped* due to early decision stop.

## 19-21. Alternatives, Selection, and Risks
- **Selected Method for Next Stage**: Implement Iteration B static fixes (mask-guided semantic roles) to resolve the factor collapse before proceeding to gradient calibration and multi-level wiring runs.
- **Unresolved Risks**: Unsupervised routing fundamentally fails without either explicit contrastive boundaries or hard assigned mask-guided roles.

## 22-26. Exact Decision-Tree Path & Final Decisions
- **Path**: Stage 0 Protocol Audit -> Iteration A Real-Data Audit -> Factor routing is uniform and unsupported by gradients/objective -> Stop Iteration A.
- **Final Decision**: `FIX_OBJECTIVE_WIRING`
- **8 Epochs Justified?**: **No**. The factor collapse must be resolved first.
- **Confirmation**: No exact medical test ran. No prohibited Git commands ran.
