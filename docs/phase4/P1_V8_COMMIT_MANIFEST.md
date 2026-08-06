# ACD-CLIP Phase4 P1-v8 Commit Manifest

## Repository Context
- **Current Branch**: `phase4-progress1-cops-dynamic-prompt`
- **Pre-commit HEAD**: `767593049d4d55fabd1e8297685d980dd1b9dd19`
- **Architecture Mode**: P1-v8 Global Hard-Anchor + CoPS Dynamic Factors + M=4 Dense Patch Routing + Bounded Rho Residual + Experts OFF.
- **Final Decision State**: `FIX_LOCAL_OBJECTIVE`

---

## Architecture Flowchart

```
hard anchor global
      |
      +--> classification + Phase2B DFG
      |
patch/prototype/VAE context
      |
factor IDs + factor heads
      |
M=4 dynamic local text factors
      |
dense patch router
      |
rho-bounded local residual
      |
final segmentation logits
```

---

## Scope of Files Included in Commit

### 1. Relevant Core Source & Architecture (`model/`, `train.py`, `test.py`, `utils.py`)
- `model/adapter.py`: P1-v8 hard-anchor global mode, CoPS dynamic factors, local rho residual, experts-off plumbing.
- `model/checkpoint_utils.py`: Phase4 H6 checkpoint saving, loading, validation, and metadata functions.
- `model/clip.py`: VisionTransformer layer feature extraction and grid-size handling.
- `model/h6/losses.py`: Delta-T diversity loss, factor orthogonal loss, center loss, functional decorrelation loss.
- `model/h6/model.py`: H6 model core, dynamic factor bank fusion, prediction routing, local h6_logit computation.
- `model/h6/router.py`: Dense patch router, local_text projection, unique topk pair diagnostics.
- `model/h6/semantic_bank.py`: Semantic bank, factor identity embeddings, slot initialization, prototype management.
- `model/h6/cluster_responsibility.py`: Tier-3 cluster responsibility target generation and loss computation.
- `model/openai.py`: OpenAI CLIP checkpoint loading utilities.
- `test.py`: P1-v8 inference evaluation and support-aware aggregation integration.
- `train.py`: P1-v8 CLI args, loss composition, structural smoke batch limits, drift diagnostics, checkpoint building.
- `utils.py`: Segmentation loss calculation and evaluation metrics.

### 2. Relevant Diagnostic, Audit, and Triage Scripts (`tools/`, `scripts/phase4/`, `tests/`)
- `tools/audit_t2_functional_wiring.py`: Audit script for Tier-2 functional loss gradient flow.
- `tools/audit_tier3_checkpoint.py`: Audit script for Tier-3 cluster responsibility and loss share.
- `tools/build_tier3_patch_bank.py`: Patch bank accumulation for Tier-3 k-means centroids.
- `tools/diagnose_smoke_factor_specialization.py`: Diagnostic probe for $\Delta T$ factor specialization and residual effects.
- `tools/eval_smoke_s0_s1.py`: S0 vs S1 triage evaluator across smoke checkpoints.
- `tools/metric_parity_harness.py`: Metric parity harness for verifying Phase2B baseline parity.
- `tools/produce_attribution.py`: Attribution table generator for A0-A4 inference modes.
- `tools/profile_runtime.py`: Runtime profiling tool for measuring R0-R3 inference latency.
- `tools/reaggregate_support_aware.py`: Support-aware macro metric aggregation tool.
- `tools/verify_dense_sparse_realimage.py`: Real-image dense vs sparse comparison probe.
- `tools/verify_dense_sparse_smoke_ep3.py`: Dense vs sparse tensor difference probe for smoke epoch 3.
- `tools/verify_local_realimage.py`: Real-image local branch spatial residual probe.
- `tools/verify_losses_and_gradients.py`: Loss decomposition and gradient norm audit probe.
- `tests/test_h6_tier3.py`: Test suite for Tier-3 cluster responsibility module.
- `scripts/phase4/run_inference_attribution_triage.sh`: Shell script for attribution triage.
- `scripts/phase4/run_p1_v8_t1a.sh`: Shell script for P1-v8 T1-A training run.
- `scripts/phase4/run_phase2b_triage.sh`: Shell script for Phase2B triage.
- `scripts/phase4/run_progress1_v8_minimal_fix.sh`: Shell script for P1-v8 minimal fix verification.
- `scripts/phase4/run_runtime_profile.sh`: Shell script for runtime latency profiling.
- `scripts/phase4/run_structural_smoke.sh`: Shell script for 3-epoch structural smoke training.
- `scripts/phase4/run_triage_A0_A3.sh`: Shell script for A0-A3 ablation triage.
- `scripts/phase4/run_wiring_smoke.sh`: Shell script for 50-batch wiring smoke.

### 3. Documentation & Evidence (`docs/phase4/`)
- `docs/phase4/P1_V8_CURRENT_ARCHITECTURE_AND_EVIDENCE.md`: Main architecture and evidence report.
- `docs/phase4/P1_V8_COMMIT_MANIFEST.md`: This commit manifest.
- `docs/phase4/evidence/p1_v8/tier1_vector_specialization_summary.json`: Compact Tier-1 evidence.
- `docs/phase4/evidence/p1_v8/tier2_wiring_audit_summary.json`: Compact Tier-2 wiring audit summary.
- `docs/phase4/evidence/p1_v8/tier2_functional_summary.json`: Compact Tier-2 functional evidence.
- `docs/phase4/evidence/p1_v8/tier3_audit_summary.json`: Compact Tier-3 audit summary.
- `docs/phase4/evidence/p1_v8/experiment_decision_tree.md`: Decision tree and matrix.

---

## Files Intentionally Excluded from Commit
- **Large Model Checkpoints / Weights**: All `.pth` files in `runs/phase2b/` and `runs/phase4/`.
- **Run Artifacts & Databases**: All directories under `runs/`, patch banks, feature banks, raw CSV outputs.
- **Unrelated Working-Tree Scratch Scripts**: `fix_getattr.py`, `generate_argparse.py`, `inject_args.py`, `inject_args_fixed.py`, `stage_c_ablation.py`, `temp_rewrite.py`, `test_shape.py`, `train_profile.py`, `tools/instrument_train.py`, `tools/phase4_v8_minimal_fix.py`.
- **Temporary Markdown Notes & Logs**: `P1_fast_audit_...md`, `TIER3_ROUTER_PATCH_CONTRACT.md`, `*.log` files.

---

## Static Validation Results
1. `python -m compileall train.py test.py model tools`: `PASSED` (0 syntax errors)
2. `git diff --check`: `PASSED` (0 trailing whitespace / conflict marker errors)
3. `python train.py --help`: `PASSED` (CLI argument parser validated)
4. `python test.py --help`: `PASSED` (CLI argument parser validated)
5. `bash -n scripts/phase4/*.sh`: `PASSED` (0 shell syntax errors)

---

## Known Unresolved Issues & Next Decision Step
- **Issue**: Unsupervised $k$-means clusters partition normal/background patches ($> 96.8\%$ normal) rather than anomaly-specific regions. Loss coefficient $\lambda_{\text{cluster\_resp}} = 0.05$ dominated training at $44\% - 52\%$ of total loss.
- **Decision State**: `FIX_LOCAL_OBJECTIVE`
- **Next Decision Step**: Implement mask-supervised patch strata (Strata 0: Normal, Strata 1: Hard-Negative, Strata 2: Boundary, Strata 3: Anomaly Core) with soft dense routing targets and calibrated loss contribution ($\le 5\%$).

---

## Manual Remote Push Instructions
*Authentication token is not provided to agent. User performs push manually.*

```bash
cd /home/ai4/caohuy/ACD-CLIP-phase4
git push origin HEAD:phase4-progress1-cops-dynamic-prompt
```

### Post-Push Verification Command
```bash
git ls-remote origin refs/heads/phase4-progress1-cops-dynamic-prompt
git rev-parse HEAD
```
*(The two commit hashes should match).*
