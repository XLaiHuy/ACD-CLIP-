# P35 Implementation Report — Soft Actionability Reweighting

Status: `P35_PASS_TO_SCIENTIFIC_PROTOCOL`

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
- `tools/sabra_v2/p35_reference.py`: readable canonical separable
  deployment-algebra reference.
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
production/reference parity, a full-target/zero-importance regression, a
cached fit-batch forward/backward/optimizer step, checkpoint save and strict
reload, a five-step microprofile, and a five-warmup/40-step profile. It loads
only frozen fit/source cache fields (`seg_features` and `teacher_region`),
with source masks and native logits disabled. It performs no CLIP, Phase2B,
teacher, held-GT, or held-mask read and no cache rebuild.

The generated engineering evidence is recorded in:

- `P35_ENGINEERING_RUN/P35_ENGINEERING_RUN.json`
- `P35_SPEED_PROFILE.json`
- `P35_ENGINEERING_QUALIFICATION.json`
- `P35_REFERENCE_PARITY_AUDIT.json`

The engineering pass completed with `P35_PASS_TO_SCIENTIFIC_PROTOCOL`.
The cached smoke changed the student (`parameter-delta L2=0.013964320754`),
kept the frozen teacher path unchanged, and strict checkpoint reload passed.
The target remained exactly the full detached teacher effect in every measured
step. Peak GPU allocated/reserved memory was `39,084,032` / `60,817,408`
bytes; peak process RSS was `1,684,544` KiB.

The five-step microprofile had median comparable step `0.004708704` s and
median end-to-end step `0.039524258` s. The warmed 40-step profile had:

| component | P35 median |
|---|---:|
| comparable step | `0.004735744` s |
| end-to-end step | `0.039106314` s |
| input/cache | `0.036353608` s |
| adapter forward | `0.000558080` s |
| objective | `0.000310272` s |
| backward/optimizer | `0.001720831` s |

Against the frozen P33 profile, the raw end-to-end comparison is `+33.23%`,
but the comparable compute path is `+0.52%` and objective-only cost is
`+0.82%`. The difference is entirely input/cache wait. A separate
read-only loader check on the same frozen cache produced median access times
of approximately `18.1–18.2 ms` for one access order and `34.5–34.8 ms` for
other orders; P35's `34.2 ms` loader median is therefore explained by the
CPU/page-cache-sensitive path documented by P33. No P35 objective or
inference path was changed to mask this distinction. Inference overhead is
`0%`.

The final status, measured profile values, memory, gradient audit, checkpoint
identity, and incident/fix record are authoritative in the generated
artifacts above. This report does not authorize P35 scientific execution.

An initial independent deployment-path reference exposed an FP32 accumulation
order mismatch in a high-scale case. This was an engineering-only reference
issue: the production objective and frozen scientific equation were unchanged.
The reference was aligned to the canonical separable deployment algebra and
the strict follow-up audit now passes on CPU and CUDA with the preregistered
`atol=1e-6`, `rtol=1e-6`: maximum component, loss, and student-gradient
absolute errors are all `0.0`. The follow-up is recorded in
`P35_REFERENCE_PARITY_AUDIT.json`.

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

No scientific formulation change was made after the preregistration freeze.
The inherited P34 report-wrapper schema defect was fixed with a pre-attempt
regression; P34 evidence and P35 semantics were unchanged. Any future
semantic change would be a preregistration-deviation stop, not an engineering
fix.
