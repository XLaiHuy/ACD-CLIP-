# P33 Implementation and Engineering Report

Status: `P33_IMPLEMENTATION_AND_ENGINEERING_QUALIFICATION_COMPLETE`

The authoritative preregistration is
[`P33_PREREGISTRATION.md`](P33_PREREGISTRATION.md), SHA-256
`d2460555be14af7d23316e43ad16c8585faeecbedf1698ee71f29dce765aed6c`.
No P33 scientific Stage 2 prediction, score, UUID, or execution marker was
created.

## 1. Exact correspondence

Production computes exactly:

```text
student_effect = D(mean_stage(student_region))
teacher_effect = stop_gradient(D(teacher_region))
weight         = stop_gradient(clamp(abs(teacher_effect)/C, 0, 1))
loss           = mean(weight * SmoothL1(student_effect, teacher_effect,
                                         beta=1, reduction=none))
```

Inputs are FP32 `[3,B,9,9]` and `[B,9,9]`; effects and weights are
`[B,518,518]`. There is one objective, no new learnable parameter, no new
tuned scalar, no auxiliary term, no teacher inference, and no target
shrinkage.

## 2. Files added

- `tools/sabra_v2/p33_objective.py` — production objective and contract;
- `tools/sabra_v2/p33_reference.py` — readable full deployment-algebra
  reference;
- `tools/sabra_v2/run_p33_engineering.py` — fit-cache-only CLI smoke,
  checkpoint reload, and speed profile;
- `tools/sabra_v2/p33_preflight.py` — deterministic synthetic/source-only
  preflight;
- `tools/sabra_v2/forensics/p33_selective_actionability.py` — frozen P32 vs
  P30R1 forensic;
- `tests/test_p33_objective.py` — focused contract, null, gradient, and
  parity tests;
- `P33_FORENSIC_ANALYSIS.json`, research decision, preflight, optimization,
  preregistration, and engineering evidence artifacts.

No P29, P30, P30R1, P31, or P32 scientific evidence or implementation was
modified.

## 3. Production/reference parity

The focused test suite passed `10` tests. It exercised four deterministic
parity cases on CPU and CUDA (eight production/reference comparisons), plus
contract, null, scale, and forbidden-path checks. Maximum absolute
production/reference differences were:

```text
loss              1.862645149230957e-09
student effect    3.0517578125e-05
teacher effect    7.152557373046875e-07
weight            1.4901161193847656e-07
student gradient  3.492459654808044e-10
```

The explicit test tolerances are `rtol=1e-5, atol=1e-4` for loss/student
effect, `rtol=1e-5, atol=1e-5` for teacher effect/weight, and
`rtol=1e-5, atol=1e-6` for the student gradient. All cases were finite and
teacher gradients were absent.

## 4. Production-path smoke

The CLI path was exercised as:

```text
CLI -> cached Tier-A/Tier-B source dataset -> RegionResidualAdapter
    -> P33 objective -> backward -> AdamW -> checkpoint -> strict reload
```

It used one cached candle fit batch, with source masks and native logits
disabled. The student parameter delta was nonzero (`L2 0.0139643369`, max
absolute `0.0009999997`); the smoke loss was finite (`0.2538076043`) and the
gradient L2 was `0.0824554488`. The checkpoint had schema
`P33_ENGINEERING_CHECKPOINT_V1` and strict state-dict reload passed.

## 5. Data and forward audit

The fit-only loader validated the frozen `/workspace/p27r1_cache_v1` source
cache, 1,962 fit records, and 200 structurally excluded held records. No
held GT/mask was read, no source mask/native-logit tensor was loaded, and no
cache was rebuilt. The engineering run performed `51` optimizer steps and
`52` adapter forwards including the reload probe. These are engineering-only
operations.

```text
new scientific Stage 2 attempts = 0
new Stage 3 attempts             = 0
full runs                        = 0
held-result tuning iterations    = 0
new CLIP forwards                = 0
new Phase2B forwards             = 0
new teacher forwards             = 0
cache rebuilds                   = 0
scientific UUIDs                 = 0
execution markers                = 0
```

## 6. Speed and memory

The warmed 40-step comparable cached median was `0.0047113920` seconds,
versus frozen P30R1 `0.0043939841` seconds: `7.224%` end-to-end comparable
overhead. Objective-only median was `0.0003077600` seconds versus
`0.0002277920`: `35.106%`. This extra objective time is attributable to the
required full-resolution absolute-value, clamp, and weighted-error elementwise
operations; it is not an unexplained data/cache stall. The comparable
end-to-end gate remains below `10%`, and the objective/input/cache components
are reported separately in `P33_SPEED_PROFILE.json`.

Against P32, the measured P33 comparable step was within run-to-run noise
(`-2.224%`), while the objective-only component was `29.26%` higher. Input
cache time is the dominant end-to-end component. Peak allocated GPU memory
was `38,010,368` bytes, `2.86%` above the frozen P32 profile and below the
10% gate; reserved memory was unchanged at `60,817,408` bytes. Incremental
inference overhead is `0%` because actionability is training-only.

## 7. Final gate

Synthetic/source-only preflight, import/compile, focused tests,
production/reference parity, cached smoke, checkpoint strict reload, 5-step
microprofile, and warmed 40-step profile passed. The only non-preferred
measurement is objective-only overhead, which is explicitly explained by the
new required operation and does not breach the end-to-end engineering gate.

`P33_PASS_TO_SCIENTIFIC_PROTOCOL`
