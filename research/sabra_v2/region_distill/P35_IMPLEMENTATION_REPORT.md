# P35 Implementation Report — Soft Actionability Reweighting

Status: `ENGINEERING_QUALIFICATION_PENDING`

P35 is the frozen, non-scientific implementation of
`SOFT_ACTIONABILITY_WEIGHTED_FUNCTIONAL_TRANSFER`. It preserves the full
signed deployed teacher effect as the target and changes only the detached,
source-only loss-importance map from P33's hard clamp to the preregistered
parameter-free `tanh` map. No P35 scientific UUID, held prediction, Stage 2,
Stage 3, or full run is created by this implementation path.

## Frozen correspondence

The authoritative preregistration is
[`P35_PREREGISTRATION.md`](P35_PREREGISTRATION.md), SHA-256
`d92a8144e071412608292b4c48f5fe69381f82c3b205f6990266f2383336e3d8`.
The implementation is exactly:

```text
E_s = deployed_margin_effect(mean_stage(student_region))
E_t = deployed_margin_effect(teacher_region)
x   = abs(stop_gradient(E_t)) / 4.960109710693359
w   = stop_gradient(tanh(x))
L   = mean(w * SmoothL1(E_s, stop_gradient(E_t), beta=1, reduction=none))
```

The target is the full `E_t`; it is never multiplied by `w`. P35 therefore
retains radial identifiability, uses one objective, and intentionally retains
P33's zero-importance semantics at exact zero actionability. P35 is not P34
target shaping, a sparsity regularizer, a learned gate, or an inference-time
module.

## Files and minimal code delta

- `tools/sabra_v2/p35_objective.py`: production objective and contract.
- `tools/sabra_v2/p35_reference.py`: readable deployment-algebra reference.
- `tools/sabra_v2/run_p35_engineering.py`: fit-cache-only smoke and profile.
- `tests/test_p35_objective.py`: objective, full-target, zero-importance, and
  production/reference parity tests.
- `tests/test_p35_preflight.py`: frozen preflight and map-contract tests.
- `tools/sabra_v2/forensics/p35_soft_actionability.py` and
  `tools/sabra_v2/p35_preflight.py`: source/synthetic evidence producers.

The existing adapter, cache format, optimizer family, checkpoint format, and
deployment effect algebra are reused. No unrelated model or historical
scientific evidence is modified.

## Verification plan and audits

The engineering runner verifies, in order, import/objective execution,
production/reference parity, a P33-versus-P35 full-target regression, a
cached fit-batch forward/backward/optimizer step, checkpoint save and strict
reload, a five-step microprofile, and a five-warmup/40-step profile. It loads
only frozen fit/source cache fields (`seg_features` and `teacher_region`),
with source masks and native logits disabled. It performs no CLIP, Phase2B,
teacher, held-GT, or held-mask read and no cache rebuild.

The generated engineering evidence is recorded in:

- `P35_ENGINEERING_RUN/P35_ENGINEERING_RUN.json`
- `P35_SPEED_PROFILE.json`
- `P35_ENGINEERING_QUALIFICATION.json`

The final status, measured profile values, memory, gradient audit, checkpoint
identity, and any incident/fix record are authoritative in those generated
artifacts. This report is updated after that engineering pass; it does not
authorize P35 scientific execution.

## Scientific integrity

P35 has zero new tuned hyperparameters, zero new learnable parameters, one
objective, zero category-specific parameters, and zero inference overhead.
Engineering optimizer steps are qualification-only. Scientific counts remain:

```text
P35 scientific Stage 2 attempts = 0
Stage 3 attempts = 0
full runs = 0
held-result tuning iterations = 0
new CLIP forwards = 0
new Phase2B forwards = 0
new teacher forwards = 0
cache rebuilds = 0
```

No scientific formulation change is permitted after the preregistration
freeze. Any required semantic change would be a preregistration-deviation
stop, not an engineering fix.
