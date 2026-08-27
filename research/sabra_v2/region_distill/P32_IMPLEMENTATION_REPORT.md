# P32 Implementation and Engineering Report

Status: `P32_IMPLEMENTATION_AND_ENGINEERING_QUALIFICATION_COMPLETE`

P32 is the frozen `FUNCTIONAL_MARGIN_EFFECT` hypothesis. The authoritative
preregistration is
`P32_PREREGISTRATION.md`, SHA-256
`5141722b2c3e3d3aac721390a8943d54356dd17bdfdad8aaa6bd7302766a5cc2`.
No P32 scientific Stage 2 prediction or score was produced during the
engineering handoff. The separate scientific runner and hard-locked cached
trainer were added afterward in a descendant commit solely to execute the
explicitly authorized one-attempt Stage 2 path; they do not change the frozen
objective or preregistration.

## 1. Exact preregistration correspondence

Production computes exactly one objective:

```text
student_effect = T(mean_stage(student_region))
teacher_effect = stop_gradient(T(teacher_region))
loss = mean SmoothL1(student_effect, teacher_effect, beta=1.0)
```

`T` is the frozen Industrial deployment transform: bilinear `9x9 -> 37x37`,
Gaussian blur with kernel `7` and sigma `1.0`, then bilinear `37x37 ->
518x518`, both with `align_corners=True`. Inputs are FP32 with shapes
`[3,B,9,9]` and `[B,9,9]`; effects are `[B,518,518]`.

There is no native-logit dependency, sigmoid in the objective, raw-direction
term, ranking term, sparsity term, gate, class-specific parameter, or teacher
inference path. The teacher is detached exactly once at the target boundary.

## 2. Files changed

- `tools/sabra_v2/p32_objective.py` — production objective and immutable
  contract constants;
- `tools/sabra_v2/p32_reference.py` — readable full deployment-algebra
  reference;
- `tools/sabra_v2/run_p32_engineering.py` — cached fit-only engineering
  smoke, checkpoint reload, and speed profile;
- `tools/sabra_v2/train_region_distill_p32_cached.py` — hard-locked P32
  scientific fit path using only the source cache;
- `tools/sabra_v2/run_p32_scientific_stage2.py` — one-attempt P32 Stage 2
  firewall, GT-free prediction freeze, post-freeze scoring, and audit path;
- `tests/test_p32_objective.py` — shape, null, scale, gradient, auxiliary-term,
  CPU/CUDA parity, and reference tests;
- `research/sabra_v2/region_distill/P32_*` — decision, preflight,
  preregistration, and engineering evidence.

No P29, P30, P30R1, P31, frozen checkpoint, or prior scientific evidence file
was modified.

## 3. Production/reference parity

The reference executes the existing `symmetric_margin_delta` and
`deploy_native_logits` path with zero native logits and extracts the deployed
abnormal-minus-normal margin. Production uses the algebraically equivalent
fixed separable matrix. Ten deterministic cases (five on CPU and five on
CUDA) passed with these explicit FP32 tolerances:

| quantity | tolerance |
|---|---:|
| loss | `rtol=1e-5`, `atol=1e-4` |
| student effect | `rtol=1e-5`, `atol=1e-4` |
| teacher effect | `rtol=1e-5`, `atol=1e-5` |
| student gradient | `rtol=1e-5`, `atol=1e-6` |

Maximum observed absolute errors were:

```text
loss              2.9802322387695312e-08
student effect    4.57763671875e-05
teacher effect    1.1920928955078125e-06
student gradient  2.3283064365386963e-09
```

The largest effect error occurred in the `100x` FP32 stress case; all cases
were within the locked tolerances and finite.

## 4. Production-path smoke

The CLI runner exercised the actual cached path:

```text
CLI -> cached Tier-A/Tier-B dataset -> RegionResidualAdapter
    -> P32 objective -> backward -> AdamW -> checkpoint -> strict reload
```

The smoke used one fit batch after the preregistration freeze. It passed with
finite loss `0.424630343914032`, gradient L2 `0.20999093353748322`, gradient
maximum absolute value `0.09342052042484283`, and a nonzero student parameter
delta (L2 `0.013964422925999368`, maximum absolute delta
`0.0009999998146668077`). The engineering checkpoint was saved and strictly
reloaded with the unchanged `RegionResidualAdapter` state-dict contract.

The smoke loaded only `seg_features` and `teacher_region`; native logits and
source masks were disabled. Teacher tensors and teacher effects did not
require gradients. No fallback to an older objective occurred.

## 5. Data and model-forward audit

The cached engineering path used the frozen `/workspace/p27r1_cache_v1`
source cache, candle LOCO fit records `1,962`, with `200` held records
structurally outside the dataset. The cache was read-only and no source mask
or native-logit field was loaded.

```text
held GT reads             = 0
held mask reads           = 0
new CLIP forwards         = 0
new Phase2B forwards      = 0
new teacher forwards      = 0
cache rebuilds            = 0
scientific held outputs   = 0
scientific UUIDs          = 0
execution markers         = 0
```

The 51 optimizer steps and 52 adapter forwards (including the strict-reload
probe) were engineering-only smoke/profile operations. They are not a
scientific training run or a held-result tuning iteration.

## 6. Speed and memory

The first direct deployment implementation was replaced by a fixed separable
operator after an engineering-only profile. The optimized implementation
computes `A @ x @ A.T` using the canonical FP32 `518x9` transform matrix; the
equation and all scientific constants are unchanged.

On the same CUDA host, the 40-step warmed comparable cached step was
`0.004818620283156634` seconds versus frozen P30R1's
`0.004393984079360962` seconds: `9.664%` overhead. Objective-only median was
`0.0002380959987640381` seconds versus `0.00022779200226068496`: `4.523%`
overhead. The 5-step microprofile was finite with median comparable step
`0.004719124868512154` seconds and objective median
`0.00023039999604225158` seconds.

DataLoader/cache-read time is reported separately: the warmed median was
`0.022011838387697935` seconds, while cache tensor transfer median was
`0.002187316073104739` seconds. The frozen P30R1 timing excludes DataLoader
wait, so this I/O-sensitive component is not misattributed to the objective.

Peak P32 CUDA allocated memory was `36,954,624` bytes and reserved memory was
`60,817,408` bytes. No duplicate teacher tensor or retained graph was found.
The small reserved-memory difference from the P30R1 profile is explained by
the required `[B,518,518]` functional-effect working tensor and CUDA allocator
behavior; allocated memory did not increase relative to the recorded P30R1
qualification (`43,625,472` bytes).

Incremental inference overhead is `0%`: P32 adds no teacher, gate, auxiliary
network, or inference branch. The objective is training-only.

## 7. Tests and incidents

The focused P29–P32 regression command passed `74` tests with `0` failures.
The only warning was the pre-existing `pkg_resources` deprecation warning.
The P32-specific suite passed `10/10`.

CUDA emitted known determinism warnings for existing reflection-padding,
adaptive-pooling, and cuBLAS backward kernels under the repository's
`warn_only=True` deterministic policy. They did not produce non-finite values
or parity failures and require no scientific change.

## 8. Scientific deviation and final gate

Scientific deviations after the preregistration hash: `0`. The separable
matrix is an implementation-only, numerically tested optimization of the
frozen linear deployment transform. No beta, normalization, threshold,
optimizer, schedule, seed, architecture, data selection, objective, or
success criterion changed.

The P32 engineering qualification is complete. A future scientific run still
requires a separate explicit authorization and must include the preregistered
native/zero-adapter control and frozen P30R1 comparator.

`P32_PASS_TO_SCIENTIFIC_PROTOCOL`
