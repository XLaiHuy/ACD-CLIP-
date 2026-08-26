# P31 Implementation Report — Native / Zero-Adapter Control

Status: `IMPLEMENTATION_COMPLETE`

## 1. Frozen contract

Authoritative preregistration:

```text
path = research/sabra_v2/region_distill/P31_PREREGISTRATION.md
sha256 = f42f0add36c0de2e303e6f25b0d48b63c33eda7d4c56d2a7ccb368ca76c865e3
```

The implementation matches `P31_NATIVE_ZERO_ADAPTER_CONTROL` exactly:

```text
R_P31(N)     = 0                  (conceptual only)
C_P31(N)     = copy(N)
Delta_P31(N) = C_P31(N) - N = 0
objective    = none
gradient     = none
```

No model, adapter, teacher, optimizer, checkpoint, or scientific prediction
path is imported or called. The selected null hypothesis is not converted into
a learned approximation.

## 2. Files changed

- `tools/sabra_v2/p31_native_control.py`: pure NumPy reference/production
  identity control, locked metric comparator, source-only cache audit,
  synthetic preflight, and offline speed-profile CLI.
- `tests/test_p31_native_control.py`: eight focused contract, parity,
  invalid-input, metric-lock, source-firewall, hash, and profile tests.
- `research/sabra_v2/region_distill/P31_*`: required optimization, preflight,
  frozen protocol, implementation, qualification, and speed artifacts.

No P29, P30, P30R1, forensic, checkpoint, or training implementation file was
modified.

## 3. Production/reference parity

The readable `reference_native_control` uses a validated NumPy copy. The
production `native_control` uses `empty_like` plus `copyto` to avoid an
unnecessary intermediate. On all 15 synthetic adversarial cases:

- output max absolute error: `0.0`;
- output L2 delta: `0.0`;
- exact output equality: `true`;
- loss: `null` because objective count is `0`;
- student gradient L2/max/nonzero fraction: `0.0` / `0.0` / `0.0`;
- finite status: `true`;
- batch-dominance flag: `false`.

The preflight artifact records every case and the frozen preregistration hash.

## 4. Static tests and regressions

Fast-fail order completed as follows:

1. import/compile: `PASS`;
2. P31 objective/identity tests: `8 passed`;
3. P30R1 forensic/execution contract regressions: `7 passed`;
4. production/reference parity and gradient contract: `PASS` through the P31
   suite; gradients are structurally zero because no objective exists;
5. source-only cached audit: `PASS`, with no held labels/masks and no model
   forward;
6. checkpoint save/reload: `NOT_APPLICABLE_NO_CHECKPOINT`;
7. 5-step micro-profile: `PASS`;
8. 40-step warmed profile: `PASS`.

The combined focused regression result was `15 passed, 1 pre-existing
pkg_resources deprecation warning`.

## 5. Exact production-path smoke

For this evaluation-only protocol, the adapted production path is:

```text
CLI
 -> p31_native_control
 -> source-only cache audit / resident-array contract
 -> native identity copy and zero-delta diagnostic
 -> JSON artifact
```

There is intentionally no cached model batch forward, backward pass,
optimizer step, checkpoint write, or reload probe. A cached model forward would
violate the frozen P31 policy. The CLI smoke and preflight exercised the actual
future offline path using synthetic and source-only arrays.

## 6. Data, model-forward, and gradient audit

Observed counts:

```text
new training runs       = 0
optimizer steps         = 0
new CLIP forwards       = 0
new Phase2B forwards    = 0
new teacher forwards    = 0
adapter forwards        = 0
cache rebuilds          = 0
held labels read        = 0
held masks read         = 0
held-result tuning      = 0
scientific markers      = 0
```

The source-only audit read only Tier-A native-logit and Tier-B teacher-region
arrays to verify finite source statistics. It did not read source masks,
held labels, held masks, or any image. The source audit now streams one array
at a time and does not retain a concatenated cache tensor.

There is no student or teacher computational graph. Consequently, all
student/frozen parameter deltas and gradients are exactly zero by construction.

## 7. Speed and memory

The profile uses resident synthetic `[3,1369,2]` FP32 arrays because P31 has no
model or cache-I/O hot path. Cache I/O is explicitly `0.0` seconds in the
profile, not silently attributed to the identity operation.

| Profile | Steps | Warmup | Median step | p90 step | Copy median | Delta median | RSS max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5-step micro | 5 | 0 | `11.592 µs` | `42.662 µs` | `5.921 µs` | `5.371 µs` | `148.27 MiB` |
| 40-step warmed | 40 | 5 | `10.230 µs` | `10.454 µs` | `4.949 µs` | `5.000 µs` | `148.27 MiB` |

Forward, objective, backward, optimizer, and input/cache time are all
`0.0` in the offline profile. New model memory is `0`, no graph is retained,
and no duplicate teacher tensor exists. Training overhead and inference
overhead are both `0%`; the preferred speed gate is met.

## 8. Second hidden-failure audit

The implementation was inspected once for missing imports, undefined symbols,
stale P30/P30R1 dispatch, incorrect CLI boundaries, output overwrite,
forbidden held access, objective fallback, shape broadcast, hidden
self-normalization, auxiliary losses, accidental forwards, duplicate launch,
and checkpoint incompatibility. The module has no model/optimizer import and
the parser exposes only `preflight` and `profile` offline commands.

The only engineering correction was replacing an exact floating-point test
assertion with `pytest.approx`; the scientific identity rule was unchanged.
The source audit was then made streaming to reduce transient memory; its
reported semantics remain finite-value/shape-only source auditing.

## 9. Scientific deviation audit

There are no scientific deviations. The authoritative Markdown hash remains
exactly:

`f42f0add36c0de2e303e6f25b0d48b63c33eda7d4c56d2a7ccb368ca76c865e3`

No loss, lambda, epsilon, normalization, threshold, metric, optimizer,
schedule, seed, architecture, source selection, or held criterion changed
after the hash was recorded.

## 10. Final gate

P31 is engineering-qualified for the future evaluation-only native-control
comparison. The future scientific comparison remains locked to native versus
the existing P30R1 output, with pAP primary, pAUROC secondary, and zero
non-inferiority margin. A failed control requires a new research decision; it
does not auto-launch a learned downstream-effect method.

```text
PASS_TO_SCIENTIFIC_PROTOCOL
```

No new scientific Stage 2, Stage 3, full, or 12-class run started.
