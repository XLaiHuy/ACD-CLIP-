# P5-E0 HRIP Evidence Audit — Adversarial Design Review

## Scope and frozen hypothesis

This is one industrial VisA TEST evidence audit. The sole primary signal is
`HRIP_SHARED_SOFT_PROJECTION`; `candidate=NONE`. It changes only the
representation of the already-frozen B1 same-image reference evidence. It
does not change Phase2B, the deployed predictor, D_rank, the B1 selector,
matching, risk population, triage budget, or any prediction.

## Source audit

- B1 selector: `tools/audit_phase5_reference_validity.py::nonlocal_peers`.
- B1 native deployment: `tools/audit_phase5_reference_validity.py::deploy_from_native`.
- Feature alignment: `tools/audit_phase5_second_evidence.py::align_features`.
- Percentile and population standard deviation: `tools/audit_phase5_hsir.py::percentile_rank,population_std`.
- Deployed score and D_rank reconstruction: `tools/audit_phase5_hsir.py::deploy_from_native` and `tools/audit_phase5_second_evidence.py::deploy_from_native_explicit`.
- Matching, top-risk selection, and triage: `tools/audit_phase5_second_evidence.py::{deterministic_matches,select_top,candidate_triage}`.
- Error objects: `tools/audit_phase5_hsir.py::{ap_contamination,pairwise_risks}`.
- Shifted control: `tools/audit_phase5_hsir.py::shifted_map` / the committed Phase5 convention `np.roll` by `(floor(H/3), floor(W/3))`.
- Model loading: `tools/audit_p4v_phase2b_readiness.py::load_model`.
- Text construction: `utils::get_phase2b_global_text_features`.
- Canonical model feature source: `model/adapter.py::ACDCLIP.forward` and `vision_text_fusion_gate_seg`.

The historical predictor helpers are not used for official construction
because they access `raw["mask"]`. The new official path accepts only an
already-preprocessed image tensor, class identity, and frozen runtime/model
objects.

## Adversarial challenges and resolutions

| Threat | Resolution |
|---|---|
| GT, mask, target, occupancy, or label leakage | Phase A parses only `class_name` and `image_path`; image-only records contain no GT fields. The official path has no dataset-record argument and never calls the historical `__getitem__`. |
| Selector drift | Copy the frozen B1 pool, median D_rank test, three stage percentile tests, Chebyshev distance `>3`, cosine descending rank, patch-index tie rule, `K=8`, and no fallback exactly. |
| Ordering or patch-index drift | Canonical order is class names sorted lexicographically, metadata order preserved within class, with a frozen ordering hash. Patch IDs are row-major `37x37`. |
| Preprocessing/checkpoint/config/text drift | Use the setup-authoritative paths and hashes; reproduce the deterministic TEST resize, bicubic RGB conversion, tensor conversion, and CLIP normalization; load the frozen checkpoint/config and Phase2B class text features. |
| Stage/alignment mismatch | Use the three `seg_tokens` stages, the authoritative patch grid, bilinear `align_corners=True` interpolation where needed, and L2 renormalization after alignment. |
| Normalization/temperature drift | Use float32 signal computation, L2-normalized descriptors, exactly 28 peer-pair distances, `tau=median(distances)`, machine-epsilon fallback only, and one shared alpha vector per query across all stages. |
| B1 centroid/invalid-reference mismatch | Reconstruct all three B1 stage centroids from the exact eight peers; invalid references emit zero evidence while preserving `valid_reference=false`. |
| Per-stage alpha accident | Alpha is computed once from shared descriptors and reused for stages 8/16/24; synthetic test T5 asserts this. |
| LOO changing the primary | LOO is diagnostic-only, recomputes seven-peer tau/alpha after removing exactly one slot, and cannot enter HRIP, gating, or evaluation. |
| Shifted-control reinterpretation | Construct HRIP first; shift only the frozen evidence map after construction; do not reselect peers or recompute alpha/tau/D_rank/matching. |
| Hidden candidate/threshold/tuning search | One fixed formula, no sweep, no candidate comparison, no threshold choice, no learned component, and frozen class-bootstrap seeds in `PROTOCOL.json`. |
| Duplicate official forward or unsafe resume | Atomic `RUN_STATE.json`, one inflight identity, atomic per-image record rename, refusal to resume with unresolved inflight state, and completed-record skip only under unchanged hashes. |
| Unbounded cache | Persist only compact per-image arrays and native logits/margins; no images, masks, full feature cache, or model weights. |
| Post-hoc primary replacement | `HRIP_raw` is permanently diagnostic-only; `HRIP` is the mean of the three within-image residual percentile ranks. |
| Artifact transport contamination | No merge/cherry-pick/fetch; only setup-materialized runtime files and committed scientific source/provenance are used. |
| Historical-source modification | Implementation is isolated in `tools/audit_phase5_p5e0_hrip.py`; protected B1/Phase2B/model/history files remain unchanged. |

## Actual blockers resolved before freeze

The fresh-host absence of `/tmp/p5_r0_run2` is accepted and recorded as
`r0_cache_available=false`, path and digest null. B1-required quantities are
reconstructed from the committed selector/evaluation semantics and the one
fresh image pass. No R0 recovery or regeneration is authorized.

The historical image dataset object reads masks, so it is not used by Phase
A. The image-only loader reproduces only its deterministic image transform
and reads only canonical image identity/path/class metadata. Post-hoc GT
loading is a separate explicit Phase C action after GT-free evidence has been
committed and verified.

No other blocker was found in the one-time preflight or source audit.

## Freeze invariants

- `K=8`, D_rank median, stage percentile threshold `<0.5`, Chebyshev radius 3.
- HRIP formula ID: `HRIP_SHARED_SOFT_PROJECTION`.
- `tau` is the median of exactly 28 unordered peer-peer shared-descriptor distances.
- Degenerate `tau <= eps(float32)` uses uniform `1/8` weights.
- `HRIP_raw` is diagnostic-only.
- Bootstrap unit is class, repetitions are 2000, seeds are frozen in `PROTOCOL.json`.
- Gates G0–G4 and their precedence are frozen before any official outcome.
- `training_steps=0`, `medical=false`, `candidate=NONE`.
