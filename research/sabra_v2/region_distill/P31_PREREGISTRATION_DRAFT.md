# P31 Preregistration Draft — Native / Zero-Adapter Control

`DRAFT_ONLY`

This document is a preregistration draft, not final execution authorization.
It contains no scientific UUID, execution marker, final preregistration hash,
trainer, runner, checkpoint, or model change.

## 1. Scientific hypothesis

**Primary hypothesis — `P31_NATIVE_ZERO_ADAPTER_CONTROL`:**

On the locked SABRA one-class comparison, the frozen native detector is
non-inferior to the P30R1 teacher-residual intervention. Therefore raw teacher
residual imitation is not necessary as a default SABRA component unless a
separate, future preregistered test demonstrates a downstream gain.

Scope is limited to the frozen comparison and data named below. This is not a
claim that native is optimal for every class or dataset.

## 2. Mechanism

The selected mechanism is no intervention, not a new learned adapter:

```text
r_P31(x) = 0
ℓ_P31(x) = ℓ_native(x)
A_P31(x) = A_native(x)
```

This is the zero-objective Route C control. It tests whether any downstream
teacher effect is necessary before considering output-effect or rank-transfer
learning. Teacher direction, sign, cosine, Pearson, Spearman, sparsity, and
teacher scale are not optimization targets.

## 3. What stays frozen

- the native CLIP/P26/Phase2B provenance and checkpoint identities;
- the existing native logits and deterministic deployment operator;
- the P30R1 frozen prediction artifacts and their provenance;
- image/sample ordering, masks used only for an already-defined deterministic
  reconstruction, resizing, interpolation, score map construction, and metric
  code;
- the inherited held-set pAP primary and pAUROC secondary evaluation
  definitions.

No P29, P30, or P30R1 scientific evidence, model, checkpoint, or training
implementation may be modified.

## 4. Allowed data

Allowed read-only inputs are:

- the terminal P30R1 forensic report and JSON;
- the frozen native/zero-adapter predictions or exact deterministic
  reconstruction already recorded by the forensic;
- the frozen P30R1 held predictions and provenance hashes;
- immutable metadata needed to verify sample identity and tensor shape;
- held labels only after this draft is locked and only for the final outcome
  calculation of the predeclared pAP/pAUROC comparison.

Held labels are forbidden for selecting the hypothesis, selecting a method,
choosing a threshold, setting a coefficient, or deciding whether a teacher
effect should be transferred. No source/held data outside the named frozen
artifacts may be added to this control.

## 5. Forbidden data and actions

- no held-label tuning or retrospective threshold selection;
- no new class, dataset, or 12-class expansion;
- no reconstruction of a new teacher target to rescue the control;
- no cache rebuild;
- no teacher forward, CLIP forward, Phase2B forward, adapter forward, or
  optimizer step;
- no modification of a scientific model, deployment operator, checkpoint, or
  existing evidence;
- no learned gate, ranking loss, L1 penalty, auxiliary network, loss weight,
  category-specific parameter, or teacher-at-inference path;
- no automatic transition to Candidate 2 or Candidate 3 if the control fails.

## 6. Allowed model forwards

`0` new model forwards are allowed.

The control consumes cached native and P30R1 outputs. If an input identity or
tensor is missing or corrupt and a forward would be needed, stop and record an
engineering/data-access failure; do not silently recompute it under this
draft.

## 7. Objective and hyperparameters

| Item | Locked value |
|---|---:|
| New training objectives | `0` |
| Optimizer steps | `0` |
| New learnable parameters | `0` |
| New hyperparameters | `0` |
| Non-inferiority margin | `0.0` absolute metric units |
| Inference overhead | `0%` |

The zero margin is fixed before evaluation and is not estimated from held
labels. No loss, coefficient, gate, temperature, pair rule, or threshold is
part of P31.

## 8. Synthetic and preflight gates

P31 has no training smoke. Before the evaluation-only comparison, verify:

1. the forensic JSON status is exactly `FORENSIC_COMPLETE`;
2. all native and P30R1 prediction identifiers, shapes, dtypes, and hashes
   match the frozen provenance;
3. the deterministic native reconstruction agrees with the recorded native
   output within the inherited reconstruction tolerance;
4. native zero intervention produces exactly zero score delta from native;
5. pAP and pAUROC are finite and computed over the identical locked sample
   set and postprocessing path;
6. the audit records zero new model forwards, zero optimizer steps, zero cache
   rebuilds, and zero scientific execution markers.

Failure of any preflight gate stops P31 without a scientific conclusion.

## 9. Engineering gates

- Worktree and branch provenance must be recorded before evaluation.
- Only the P31 decision/draft artifacts may be new in this phase.
- Existing P29/P30/P30R1 files and frozen checkpoints must remain unchanged.
- No training process, scientific runner, or execution marker may be
  launched.
- The evaluation must be deterministic and use the same score-map and metric
  implementation for native and P30R1.
- Any missing/corrupt frozen input is an engineering/data-access stop, not a
  reason to create a new forward or substitute data.

## 10. Scientific Stage 2 criteria

P31 Stage 2 is evaluation-only; it contains no learned Stage 2 candle.

Define the locked paired differences:

```text
ΔpAP    = pAP(native zero-adapter) − pAP(P30R1)
ΔpAUROC = pAUROC(native zero-adapter) − pAUROC(P30R1)
```

Primary criterion: pAP. Secondary criterion: pAUROC.

The native-control hypothesis is supported for this scope when:

```text
ΔpAP ≥ 0 and ΔpAUROC ≥ 0
```

The native-control hypothesis is falsified for this scope when either locked
comparison is negative. This is a deterministic paired comparison, not a
held-label tuning exercise and not a license to alter the margin, threshold,
or score transformation.

If native is non-inferior, stop and retain the native control as the strongest
next scientific conclusion. If P30R1 wins, stop and mark the null control
falsified; a separate research decision is required before any functional
effect method is specified or trained.

## 11. Scientific stop criteria

Stop with no scientific conclusion if there is any provenance mismatch,
invalid metric, forbidden data read, new model forward, cache rebuild, or
execution marker.

Stop with the native-control conclusion if both locked differences are
nonnegative. Do not continue to a learned method merely to preserve protocol
numbering.

Stop with the control falsified if either locked difference is negative. Do not
automatically start `DOWNSTREAM_LOGIT_EFFECT_TRANSFER`, ranking transfer, a
new P30 variant, Stage 2 training, Stage 3, or a 12-class run.

## 12. No-rerun rule

There is one evaluation of the locked cached comparison. An unfavorable result
does not permit a rerun with changed thresholds, score transforms, sample
subsets, coefficients, losses, or data. A repeat is allowed only to resolve a
documented integrity/reproducibility failure using the identical frozen inputs
and code; if the scientific inputs or question change, create a new
preregistration and new identifier.

Held labels may not be revisited to select a replacement method after the
result. A failed control is a research result, not an automatic queue.

## 13. Cost and authorization

```text
new training runs       = 0
optimizer steps         = 0
new CLIP forwards       = 0
new Phase2B forwards    = 0
new teacher forwards    = 0
Stage 2 runs            = 0
Stage 3 runs            = 0
full 12-class runs      = 0
new inference cost      = 0%
```

Training required: `NO`.

This draft does not authorize execution. It does not create a scientific
marker or final preregistration hash.

`P31_PREREGISTRATION_DRAFT_ONLY`
