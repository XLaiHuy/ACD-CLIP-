# P34 Implementation and Engineering Report

Status: `P34_PASS_TO_SCIENTIFIC_PROTOCOL`

P34 replaces P33 loss-importance weighting with the preregistered explicit
actionability-shaped target. The implementation was completed after the
authoritative preregistration was frozen at SHA-256
`b78f69487e665b62d9c81b58da45f8f0afe5d047e91996f18569c6d38f99abdb`.

## 1. Exact correspondence to the frozen protocol

The production objective is:

```text
E_s = D(mean_stage(student_region))
E_t = D(detach(teacher_region))
w   = detach(clamp(abs(E_t)/4.960109710693359, 0, 1))
T   = detach(w * E_t)
L   = mean(SmoothL1(E_s, T, beta=1.0, reduction=none))
```

There is one objective and no auxiliary term. The target, weight, and teacher
path are detached; the student is not self-normalized. The deployment
operator and tensor shapes are unchanged. No hard gate, sparsity loss,
learned gate, category parameter, teacher inference, or inference branch was
added.

## 2. Files changed

- `tools/sabra_v2/p34_objective.py`: vectorized production objective and
  immutable contract.
- `tools/sabra_v2/p34_reference.py`: readable reference implementation using
  the full deployment algebra.
- `tools/sabra_v2/run_p34_engineering.py`: fit-cache-only smoke, checkpoint,
  reload, and speed qualification path.
- `tools/sabra_v2/p34_preflight.py`: source/synthetic preflight analysis.
- `tests/test_p34_objective.py`: objective, gradient, radial, parity, and
  cache-plumbing tests.
- P34 research, preflight, preregistration, qualification, and speed evidence
  under `research/sabra_v2/region_distill/`.

P29, P30, P30R1, P31, P32, and P33 scientific evidence and implementation
were not modified.

## 3. Causal regression

The explicit P33-versus-P34 regression uses effect-space tensors with
`w=0`, `E_t!=0`, and `E_s!=0` to isolate the operator difference:

- P33 `w*SmoothL1(E_s,E_t)` has exactly zero student gradient.
- P34 `SmoothL1(E_s,w*E_t)` has a nonzero gradient whose descent direction
  reduces `E_s` toward zero.

The actual source-derived rule normally has `E_t=0` whenever `w=0`; the
decoupled test is an algebra isolation, while the production test separately
verifies a zero teacher effect gives an explicit zero target and a restoring
student gradient. At `w=1`, P34 reduces to the ordinary functional transfer
target. Student magnitude remains identifiable because no student
normalization is present.

## 4. Tests and parity

- P34 objective and engineering tests: `18 passed`.
- Inherited P32/P33 objective regressions: `20 passed`.
- Production/reference cases: normal, zero, near-zero, and sign-reversed
  FP32 tensors.
- Maximum production/reference errors from the qualification run:

```text
student effect  4.76837158203125e-07
teacher effect  7.152557373046875e-07
actionability   1.4901161193847656e-07
target          5.364418029785156e-07
loss            0.0
gradient        9.313225746154785e-10
```

All are within the preregistered `1e-6` FP32 tolerances. Heavy-tail,
mixed-batch, all-abstain, all-active, scale, sign, and outlier cases were
covered by the P34 preflight and tests; all produced finite loss/gradient
behavior.

## 5. Cached production-path smoke

The exact engineering path was exercised as:

```text
cached fit batch -> RegionResidualAdapter -> P34 target/objective
-> backward -> AdamW -> checkpoint -> strict state-dict reload
```

The smoke used only 1,962 fit/source records from the locked candle cache,
with `source_mask_loaded=false` and `native_logits_loaded=false`. It
produced finite loss and gradients, changed the student, kept the frozen
teacher path unchanged, wrote `p34_engineering_adapter.pt`, and strictly
reloaded it.

Engineering-only optimizer steps: `51`. Scientific optimizer steps: `0`.
The smoke audit recorded `0` new CLIP forwards, `0` new Phase2B forwards,
`0` teacher forwards, `0` held GT/mask reads, and `0` cache rebuilds.

## 6. Speed qualification

The warmed 40-step profile measured:

| quantity | P34 |
|---|---:|
| median comparable step | 0.004566592 s |
| p90 comparable step | 0.005799024 s |
| mean comparable step | 0.004746025 s |
| median end-to-end step | 0.007434672 s |
| objective median | 0.000281184 s |
| input/cache median | 0.004758974 s |

Against P33, P34 was `-3.07%` on the comparable step and `-8.64%` on the
objective. Against P32, it was `-5.23%` on the comparable step and
`+18.10%` on the objective. The objective is reported separately because
the cached input/data-loader path is the dominant component and historical
profile schemas did not all record end-to-end time consistently. Inference
overhead is `0%`.

## 7. Memory and incident audit

The warmed run reported peak GPU allocation `45,727,744` bytes and peak RSS
`1,858,184 KiB`. A same-process comparison showed the P34 peak is transient:
the current allocation after 51 steps returned to `18,063,872` bytes, matching
the P33 current allocation, with `retained_graph=false` and
`duplicate_teacher_tensor=false`. A one-step isolated P34/P33 probe had the
same `33,942,528` byte peak; the repeated-profile peak reflects the required
detached full-resolution target and CUDA workspace/allocator behavior rather
than accumulation. RSS growth relative to P33 was `8.37%`, below the `10%`
gate. The fixed target tensor is documented; no approximation or precision
change was used.

The first qualification invocation failed before artifact writing because
cache metadata fields were incorrectly rejected. The generated engineering
checkpoint and traceback were preserved under
`P34_ENGINEERING_FAILURES/baseline_lookup_failure/`; the smallest fix
allowed the four known metadata fields and added a regression. The second
failure was an eager legacy timing-key lookup after profiles had completed; it
was fixed with a guarded key lookup and regression. Neither incident touched
P34 math, preregistration, held data, or scientific execution.

## 8. Final gate

All required implementation, preflight, parity, smoke, checkpoint, speed,
memory, and data-access gates pass. No scientific semantic deviation was
made after preregistration freeze. P34 is ready for one separately authorized
future scientific Stage 2 attempt, but no such attempt has started.

`P34_PASS_TO_SCIENTIFIC_PROTOCOL`
