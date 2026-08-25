# P25R3 Exact Q1 Numerical Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover all P25R2 Q1 folds using an exact preconditioned optimizer and an original-beta optimality certificate, then preserve frozen Q1/Q2 routing.

**Architecture:** A new `patch_actionability_r3.py` imports immutable P25R2 feature/metric/policy functions, replaces only ranker fitting, reads immutable P25R2 targets, and writes isolated P25R3 artifacts. The optimizer exposes pure beta-space objective/gradient functions plus a diagonal source-design reparameterization.

**Tech Stack:** Python 3.11, NumPy float64, deterministic damped Newton, PyTorch only through inherited frozen feature construction, pytest, Git.

**Spec:** `research/sabra_cure/patch_actionability_r3/P25R3_PREREGISTRATION.md`

## Global Constraints

- Preserve every P25R2 scientific definition and undefined-correlation semantics.
- Reuse target artifacts only after exact hash/schema/alignment audit.
- No held target may influence a fold's fit or numerical configuration.
- No marker before published clean execution base; no changes after marker.
- MVTec=0, Medical=0, additional CLIP=0, Phase2B=0.

---

### Task 1: Pure exact objective and preconditioner

**Files:**
- Create: `tests/test_sabra_cure_patch_actionability_r3.py`
- Create: `tools/sabra_cure/patch_actionability_r3.py`

**Interfaces:**
- `prepare_problem(x, groups) -> PairProblem`
- `beta_objective_gradient(beta, problem) -> tuple[float, ndarray]`
- `transformed_objective_gradient(z, problem) -> tuple[float, ndarray]`

- [ ] Write tests with hand-derived nonzero beta objective and chain-rule gradient values, plus inactive columns.
- [ ] Run the focused tests and verify missing APIs fail.
- [ ] Implement deterministic max-abs scaling, exact beta-space loss/L2, and transformed gradient.

```python
column_max = np.max(np.abs(design), axis=0)
scale = np.maximum(column_max, 1.0)
active = column_max > 0.0
beta[active] = z / scale[active]
margin = design @ beta
loss = np.mean(weight * np.logaddexp(0.0, -margin)) + 0.5 * (beta @ beta)
g_beta = np.mean((-weight * expit(-margin))[:, None] * design, axis=0) + beta
g_z = g_beta[active] / scale[active]
```

- [ ] Run focused tests and verify objective/gradient parity at `<=1e-12` relative tolerance.

### Task 2: Certified recovered solver

**Files:**
- Modify: `tools/sabra_cure/patch_actionability_r3.py`
- Modify: `tests/test_sabra_cure_patch_actionability_r3.py`

**Interfaces:**
- `fit_ranker_exact(x, groups) -> dict`
- `validate_fit(model, problem) -> dict`

- [ ] Write failing tests showing historical zero beta with nonzero gradient is rejected and an ill-conditioned fixture produces a certified nonzero fit.
- [ ] Run focused tests and verify the expected certificate failures.
- [ ] Implement the single frozen damped-Newton solver and persist all diagnostics.

```python
z = np.zeros(problem.active_count, dtype=np.float64)
for iteration in range(50):
    direction = np.linalg.solve(exact_hessian_z(z), -gradient_z(z))
    step = armijo_halving_step(z, direction, c1=1e-4, minimum=2**-30)
    z += step * direction
    if original_beta_certificate(z).valid:
        break
```

- [ ] Verify nonzero-beta recovery, constant-column handling, strict JSON-safe diagnostics, and deterministic repeat parity.

### Task 3: Target reuse and frozen Q1/Q2 controller

**Files:**
- Modify: `tools/sabra_cure/patch_actionability_r3.py`
- Modify: `tests/test_sabra_cure_patch_actionability_r3.py`

**Interfaces:**
- `audit_targets() -> dict`
- `q1_fold(held, shards, out) -> dict`
- `execute_once(out) -> dict`

- [ ] Write failing tests for target hash/schema/order rejection, 12-fold Q1 routing, null-correlation preservation, conditional Q2, and no-marker rehearsal.
- [ ] Run tests and confirm missing controller behavior fails.
- [ ] Implement P25R2 target reads, recovered Q1/Q2 wrappers, atomic progress/failure/summary artifacts, and one-marker guard.

```python
if marker.exists() or summary.exists():
    raise RuntimeError("P25R3_ENGINEERING_STOP attempt already exists")
for held in r1.CLASSES:
    fold = q1_fold(held, shards, output)
    if not fold["model"]["diagnostics"]["valid"]:
        raise RuntimeError("P25R3_ENGINEERING_STOP invalid numerical fit")
q1 = p25r2.evaluate_q1(folds)
if not q1["pass"]:
    return terminal_q1_stop(q1)
return execute_frozen_q2(folds)
```

- [ ] Run focused and inherited P25R2 tests.

### Task 4: Pre-marker audits and publication

**Files:**
- Create: `results/sabra_cure/patch_actionability_r3/target_reuse_audit.json`
- Create: `results/sabra_cure/patch_actionability_r3/numerical_parity.json`
- Create: `results/sabra_cure/patch_actionability_r3/known_failure_regression.json`
- Create: `results/sabra_cure/patch_actionability_r3/performance_audit.json`
- Create: `results/sabra_cure/patch_actionability_r3/pre_execution_audit.json`

- [ ] Run objective/gradient, synthetic ill-conditioning, inactive-column, strict JSON, routing, target, firewall, and determinism tests.
- [ ] Run the source-only chewinggum regression without reading chewinggum targets; require zero beta rejection and a certified recovered fit.
- [ ] Benchmark the complete source-only Q1 fit path and record seconds/fold, projected total, and peak RSS.
- [ ] Stage only explicit runner/test/audit paths, inspect diff, commit execution base, push, and verify clean remote equality.

```bash
pytest -q tests/test_sabra_cure_patch_actionability_r3.py tests/test_sabra_cure_patch_actionability_r2.py
git diff --check
git add -- tools/sabra_cure/patch_actionability_r3.py tests/test_sabra_cure_patch_actionability_r3.py results/sabra_cure/patch_actionability_r3/pre_execution_audit.json
```

### Task 5: Exactly one recovered attempt and terminal audit

**Files:**
- Create only after execution-base publication: `results/sabra_cure/patch_actionability_r3/ATTEMPT_STARTED.json`
- Create: isolated Q1/Q2 fold, parameter, progress, summary, post-audit, and final-decision artifacts.

- [ ] Create one UUID marker and execute all 12 Q1 folds without early interpretation.
- [ ] Validate 12/12 numerical certificates and apply inherited Q1 gates.
- [ ] Enter Q2 only when the frozen Q1 routing permits; otherwise stop scientifically.
- [ ] Independently recompute persisted numerical diagnostics and metrics, publish terminal evidence, verify remote equality and clean worktree, then stop.

```bash
python tools/sabra_cure/patch_actionability_r3.py --run-once
python tools/sabra_cure/patch_actionability_r3.py --post-audit
```
