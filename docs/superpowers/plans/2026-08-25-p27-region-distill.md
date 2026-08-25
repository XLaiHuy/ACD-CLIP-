# P27 Region Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a frozen-P26, source-only 9x9 region residual adapter and a training-ready LOCO execution base without running scientific training.

**Architecture:** `RegionResidualAdapter` consumes frozen three-stage Phase2B segmentation features, creates three 9x9 margin residual maps, and integrates them symmetrically into native logits before existing deployment. R0 teacher generation is source-GT-only and uses the historical signed action semantics with alpha 0.25.

**Tech Stack:** Python 3.12, PyTorch 2.5.1+cu121, torchvision, existing Phase2B runtime, pytest.

**Spec:** `research/sabra_v2/region_distill/P27_ARCHITECTURE.md`

## Global Constraints

- Keep P26 files, P26 behavior, and P26 checkpoint parameters immutable.
- Use exactly a 37x37-to-9x9 adaptive-average region mapping.
- Train only `RegionResidualAdapter`; use equal SmoothL1 and canonical focal/dice loss weights.
- Require VisA LOCO inventories; held class cannot reach teacher or fit paths.
- Never read Medical or MVTec; no full scientific training run in this session.

---

### Task 1: Region geometry and symmetric logit integration

**Files:**
- Create: `tools/sabra_v2/region_pool.py`
- Create: `tests/test_sabra_v2_region_pool.py`

**Interfaces:**
- Produces `pool_patch_map`, `upsample_region_map`, `symmetric_margin_delta`.
- Accepts `[B,37,37]` or `[S,B,37,37]` maps and `[S,B,1369,2]` logits.

- [ ] Write failing shape, deterministic, and margin-semantics tests.
- [ ] Run `pytest -q tests/test_sabra_v2_region_pool.py` and confirm failure.
- [ ] Implement adaptive pooling, bilinear upsampling, and two-class integration.
- [ ] Re-run the focused tests and commit the tested utility.

### Task 2: Teacher and small region adapter

**Files:**
- Create: `tools/sabra_v2/correction_teacher.py`
- Create: `tools/sabra_v2/region_adapter.py`
- Create: `tests/test_sabra_v2_teacher_adapter.py`

**Interfaces:**
- `r0_teacher_patch_delta(native_logits, masks)` returns `[B,37,37]` source target.
- `build_region_teacher(...)` returns `[B,9,9]`.
- `RegionResidualAdapter.forward(features)` returns `[3,B,9,9]`.

- [ ] Write controlled positive/negative teacher, adapter shape, zero-output, and serialization tests.
- [ ] Run the tests to confirm missing symbols fail.
- [ ] Implement R0 utility/action semantics and the fixed convolutional adapter.
- [ ] Re-run focused tests and commit the tested components.

### Task 3: Frozen Phase2B student forward and source-only LOCO inventory

**Files:**
- Create: `tools/sabra_v2/student_forward.py`
- Create: `tools/sabra_v2/data_protocol.py`
- Create: `tests/test_sabra_v2_student_forward.py`

**Interfaces:**
- `forward_region_student(adapter, seg_features, native_logits)` returns corrected logits and region residuals.
- `loco_inventory(rows, held_class)` returns disjoint fit/held records.
- `assert_frozen_phase2b(model, adapter)` validates gradient ownership.

- [ ] Write failing parity, held-class exclusion, frozen-parameter, and backward tests.
- [ ] Run focused tests and confirm failure.
- [ ] Implement frozen interface and LOCO filtering without target data paths.
- [ ] Re-run focused tests and commit.

### Task 4: Training, evaluation, and pre-training audit entrypoints

**Files:**
- Create: `tools/sabra_v2/train_region_distill.py`
- Create: `tools/sabra_v2/evaluate_region_distill.py`
- Create: `tools/sabra_v2/audit_region_distill.py`
- Create: `tests/test_sabra_v2_audit.py`

**Interfaces:**
- Train accepts one `--held-class`, `--visa-root`, `--output`, and frozen P26 assets.
- Audit emits machine-readable JSON and Markdown, with no external reads.

- [ ] Write failing config, audit, and parser tests.
- [ ] Run focused tests and confirm failure.
- [ ] Implement one-fold orchestration, held-only prediction evaluation, and audit checks.
- [ ] Re-run all P27 tests and commit.

### Task 5: Engineering smoke and handoff

**Files:**
- Create: `research/sabra_v2/region_distill/P27_TRAINING_HANDOFF.md`
- Modify: `research/sabra_v2/region_distill/P27_PROTOCOL.json`

- [ ] Run P26 regression tests, P27 tests, audit, and one tiny source GPU forward/backward/save-reload smoke.
- [ ] Record engineering-only resource observations and syntactically validate handoff commands.
- [ ] Inspect status and staged diff, commit explicit paths, push, and verify remote equals local.
