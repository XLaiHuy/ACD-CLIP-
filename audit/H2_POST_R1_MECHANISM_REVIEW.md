# H2 post-R1 mechanism review

`REVIEW_STATUS=COMPLETE_PENDING_USER_APPROVAL`

`REVIEW_SCOPE=SOURCE_EVIDENCE_AND_MECHANISM_SELECTION_ONLY`

`NEW_TRAINING_RUN=NO`

`OPTIMIZER_STEP_USED=NO`

`MEDICAL_INFERENCE_RUN=NO`

`MVTEC_INFERENCE_RUN=NO`

`TARGET_TUNING_USED=NO`

## Durable starting state

The branch and remote were fetched and verified before this review:

| check | result |
|---|---|
| branch | `research/h2-functional-anchor-bounded-r1` |
| local HEAD | `bf7f34a02e6c61686a123e245a8824398d87c0de` |
| remote HEAD | `bf7f34a02e6c61686a123e245a8824398d87c0de` |
| worktree before this review | clean |
| diagnostic map | phases 0--18 `COMPLETE_VALID` |

No diagnostic phase was rerun. The existing map is treated as authoritative;
its raw-array limitation is retained rather than silently upgraded.

## R1 result and causal reading

The matched R1 comparison is a valid mechanism result even though its control
violated the preregistered numerical skip rule. Both arms saw the same 500
transformed batches. The control had 498 successful attempts and two isolated
non-finite-gradient skips; the functional candidate had 500 successful
attempts. The candidate therefore has interpretable endpoint evidence, but the
pair is not a numerically passing training screen.

The candidate moved the intended geometry mediator in the desired direction:

| endpoint | control `A_SHORT_R1` | candidate `A_FUNC_SHORT_R1` | candidate minus control |
|---|---:|---:|---:|
| stage-2 cosine to E1 | `0.967000` | `0.996287` | `+0.029286` |
| stage-3 cosine to E1 | `0.953154` | `0.988746` | `+0.035591` |
| final AUROC | `0.891814` | `0.783803` | `-0.108011` |
| final AP | `0.320646` | `0.303131` | `-0.017515` |
| positive mean | `0.127369` | `0.089658` | `-0.037711` |
| interior mean | `0.156930` | `0.118421` | `-0.038509` |
| stage-1 AP | `0.286154` | `0.285685` | `-0.000469` |

Stage-2 AP fell from `0.274045` to `0.255508`, and stage-3 AP fell from
`0.217285` to `0.194565`. The positive and interior medians also fell. This
is not a global score rescaling: the candidate reduced background scores too,
but it reduced the bulk of positive and interior anomaly response more
strongly. The result is therefore:

`R1_MECHANISM_FAILURE=CONFIRMED`

`BOTTLENECK_DIAGNOSIS_INVALIDATED=NO`

The intervention failed; the source bottleneck remains supported. The
candidate is not authorized for lambda tuning, a horizon change, a seed retry,
or a second functional-anchor screen.

## Mechanism post-mortem

Let a late-stage token be decomposed as

```text
z_late(x) = n(x) + r_native(x) + r_anomaly(x)
```
where `n` is transferable/native structure, `r_native` is task adaptation that
remains useful across source images, and `r_anomaly` is an anomaly-
discriminative residual. The desired operation is selective preservation of
the first two terms while leaving `r_anomaly` available to the segmentation
and ranking losses. A token-wise normalized cosine penalty against frozen E1
does not have access to that decomposition.

For every stage-2/3 token, the R1 loss penalized angular deviation from E1.
Therefore a useful anomaly direction is indistinguishable from harmful drift
unless it happens to be parallel to the E1 token. The gradient cap limited the
functional term's effective norm, but it did not change its direction: every
active functional update still favored unconditional return-to-E1 geometry.
The cap consequently makes the intervention bounded, not selective.

The measured pattern is exactly the expected signature of that failure mode:

1. late-stage cosine to E1 increased substantially;
2. stage-2/3 ranking weakened rather than recovered;
3. positive and interior coverage collapsed while the ordinary negative tail
   also became smaller;
4. stage-1 health did not improve enough to compensate.

The earlier full-source audit supplies complementary evidence. A's stage-2/3
pixel AP is lower than H's, interior coverage is the most damaged region, and
Conv-LoRA accounts for `62.58%` of A's parameter-drift energy. The source
gradient probe also found abnormal Dice to be the dominant descriptive
weighted norm (`79.18%`) and found moderate, not severe, task conflicts.
These observations support late-stage overadaptation and anomaly undercoverage
as the primary source mechanism, but they do not prove that a particular
parameter family or loss pair is the unique cause.

The R1 result also distinguishes two preservation targets:

* parameter/reference preservation can be useful as a weak trust region;
* feature-value preservation at all late-stage tokens can erase the residual
  that the anomaly objective needs.

The existing Safe Anchor remains in both R1 arms and already supplies the
parameter-side family budget. Adding another unconditional feature-value
anchor is therefore the wrong direct replacement. A useful replacement must
preserve stage-1 and anomaly/interior response, allow late-stage residuals to
remain free, and control harmful late-stage influence at a selective point.

## Evidence and literature review

The existing repository evidence is stronger for a stage-contribution problem
than for a causal gradient-conflict intervention. In the frozen local
counterfactual, a small stage-1 emphasis raised subset AP from `0.361343` to
`0.364007`; emphasizing stage 2 or stage 3 lowered it to `0.360222` and
`0.359180`. Beta and tau perturbations were nearly inert. This is a cheap,
source-only directional result, not target tuning and not proof of transfer.

The literature is used only to constrain mechanism plausibility:

* L2-SP formalizes a simple reference-parameter penalty and reports that it
  preserves pretrained feature roles at low additional training cost. That
  supports a weak trust-region intuition, but R1 shows why matching late
  feature values is not sufficient for anomaly localization. See [Li et al.,
  2018](https://proceedings.mlr.press/v80/li18a.html).
* PCGrad defines projection of conflicting task gradients and is a credible
  option when conflict is demonstrated. The local H2 evidence is only
  moderate/possible, so this supports candidate #2 as a falsifiable fallback,
  not as the selected mechanism. See [Yu et al.,
  2020](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf).
* Multi-level anomaly-localization methods explicitly retain residual and
  multi-scale information rather than forcing every representation to match a
  single normal prototype. This is conceptual support for preserving a
  reliable early-stage contribution while leaving anomaly residuals available,
  not evidence that those methods transfer to this implementation. See
  [Prototypical Residual Networks](https://arxiv.org/abs/2212.02031) and
  [MFFA](https://journals.sagepub.com/doi/abs/10.3233/JIFS-222595).

No source or target paper establishes the exact proposed coefficient for this
model. The coefficient is consequently fixed by a deterministic source-only
rule in the protocol below, with no sweep and no target observation.

## Candidate set and red-team

Exactly three candidates were considered. The failed R1 mechanism is retained
in the table for audit continuity and is not an actionable next experiment.

### Candidate 1 — stage-2/stage-3 frozen-E1 functional feature anchor

`STATUS=FAILED_R1_DO_NOT_RETRY`

1. **Measured failure addressed:** stage-2/3 feature drift toward a frozen E1
   reference.
2. **Anomaly coverage:** fails this requirement in the observed screen;
   positive/interior response and stage-2/3 ranking fell.
3. **Causal position:** downstream representation intervention at the diagnosed
   mediator.
4. **Inference cost:** no new parameters or inference compute.
5. **Simplicity:** mathematically simple, but it imposes a global constraint
   on every late-stage token.
6. **Source-fit risk:** high; E1 cosine can improve while source anomaly
   ranking worsens.
7. **CLIP geometry risk:** confirmed in the relevant sense; it can preserve
   transferable geometry at the cost of anomaly-discriminative directions.
8. **Safe Anchor interaction:** compatible in code, but redundant in its
   unconditional preservation intent.
9. **DFG/SS2D interaction:** does not alter DFG or SS2D directly, but it
   constrains their produced late-stage features.
10. **Novelty:** a training-only feature preservation term is defensible as a
    bounded intervention, not as a successful method claim.
11. **Falsification in <=500 attempts:** the existing R1 already falsifies it:
    geometry improved but ranking and anomaly coverage degraded.

### Candidate 2 — family-scoped abnormal-Dice/classification conflict projection

`STATUS=NOT_SELECTED_POSSIBLE_ONLY`

1. **Measured failure addressed:** possible upstream segmentation/classification
   interference, especially classification versus abnormal Dice in Conv-LoRA
   (mean cosine `-0.322`) and abnormal-Dice norm dominance.
2. **Anomaly coverage:** could preserve anomaly residuals by removing only a
   conflicting gradient component, but it could also remove useful abnormal
   segmentation signal; this is not established.
3. **Causal position:** upstream of parameter drift and representation change.
4. **Inference cost:** none; training adds per-term gradient computation and
   projection bookkeeping.
5. **Simplicity:** less simple than fixed stage fusion and materially more
   invasive in the optimizer path.
6. **Source-fit risk:** high; it can improve the observed source task balance
   without improving transfer.
7. **CLIP geometry risk:** unresolved; protecting classification does not
   guarantee preservation of native geometry or anomaly recall.
8. **Safe Anchor interaction:** technically compatible if projection precedes
   the existing family-safe anchor budget, but the combined update rule needs
   a new parity audit.
9. **DFG/SS2D interaction:** no direct architectural conflict, but shared
   family gradients include DFG-related parameters and must be scoped exactly.
10. **Novelty:** a family-scoped PCGrad-like rule is defensible as a local
    intervention, not as a new general optimization principle.
11. **Falsification in <=500 attempts:** fail if the measured conflict does not
    decrease, if positive/interior coverage is not preserved, or if the
    candidate has unequal valid steps or new numerical failures.

The reason it is not selected is evidential, not conceptual: the H2 conflict
diagnosis is `POSSIBLE`, no severe conflict rule was met, and complete
per-term gradients are unavailable in the existing audit.

### Candidate 3 — fixed E1-calibrated reliability-residual stage fusion

`STATUS=SELECTED_FOR_ONE_BOUNDED_SCREEN`

This is a deliberately simpler refinement of the frozen
`BOUNDED_RELIABILITY_RESIDUAL_STAGE_FUSION` idea. It uses no learned gate and
no new architecture. The candidate replaces equal stage averaging with a
fixed convex stage mixture; all stage maps and their anomaly residuals remain
available.

1. **Measured failure addressed:** harmful late-stage contribution and the
   observed positive/interior undercoverage in the final fused map.
2. **Anomaly coverage:** stage 1 remains fully represented and stage 2/3
   residuals remain present at reduced weight; nothing is forced toward E1.
3. **Causal position:** downstream of late-stage representation drift. It is
   a direct control of the diagnosed mediator's harmful output influence, not
   a claim to repair every upstream parameter change.
4. **Inference cost:** three scalar multiplies and an add; zero new parameters,
   zero additional feature extraction, and no target-dependent branch.
5. **Simplicity:** simpler than a learned reliability gate and simpler than
   gradient projection; one fixed convex mixture is the only difference.
6. **Source-fit risk:** real; the source stage-AP calibration could favor VisA.
   This is why calibration is frozen before training and target data are
   excluded, and why the bounded screen requires coverage gates rather than
   E1 cosine alone.
7. **CLIP geometry risk:** it does not force feature geometry toward E1, but
   it can underweight useful late semantic information. The bounded residual
   and stage-ranking gates test that risk.
8. **Safe Anchor interaction:** compatible; Safe Anchor remains unchanged and
   no second feature anchor is added.
9. **DFG/SS2D interaction:** compatible; DFG/SS2D produce the same three
   native stage maps and their internal routing is untouched.
10. **Novelty:** defensible only as a bounded, source-calibrated reliability
    residual for this multi-stage ACD-CLIP path. No broad novelty claim is
    made for weighted multi-scale fusion itself.
11. **Falsification in <=500 attempts:** fail if late-stage contribution is
    not reduced, or if final AP/AUROC and positive/interior coverage are not
    preserved, regardless of any E1-cosine increase.

`MECHANISM_FIT=STRONG`

`EVIDENCE_SUPPORT=LIKELY_ENOUGH_FOR_BOUNDED_TEST`

## Selection

`SELECTED_NEXT_MECHANISM=FIXED_E1_CALIBRATED_RELIABILITY_RESIDUAL_STAGE_FUSION`

This is selected because it is the only remaining candidate with a direct
source counterfactual in the measured failure direction and it preserves the
desired decomposition: native/transferable structure plus anomaly residual.
It is downstream, so it must be treated as a bounded mechanism screen rather
than proof that Conv-LoRA or the late feature representation is repaired.

`NEXT_BOUNDED_EXPERIMENT_JUSTIFIED=YES`

The selected protocol is recorded separately in
`audit/H2_NEXT_BOUNDED_PROTOCOL.md` and `.json`. It is preregistration only;
user approval is required before any runner is invoked.

## Final decision fields

```text
REMOTE_SYNC=PASS
R1_MECHANISM_FAILURE=CONFIRMED
R1_FAILURE_EXPLANATION=Unconditional normalized stage-2/3 token-cosine anchoring restored E1 geometry but suppressed anomaly-discriminative residual directions, lowering stage-2/3 ranking and positive/interior coverage; the control also failed the numerical skip gate.
PRIMARY_BOTTLENECK=LATE_STAGE_FUNCTIONAL_OVERADAPTATION_AND_ANOMALY_UNDERCOVERAGE
PRIMARY_CONFIDENCE=LIKELY_SOURCE_POSSIBLE_TARGET
SECONDARY_BOTTLENECK=SEGMENTATION_GRADIENT_IMBALANCE_AND_CONFLICT
SECONDARY_CONFIDENCE=POSSIBLE
BOTTLENECK_DIAGNOSIS_INVALIDATED=NO
TOP1_NEXT=FIXED_E1_CALIBRATED_RELIABILITY_RESIDUAL_STAGE_FUSION
TOP2_NEXT=FAMILY_SCOPED_ABNORMAL_DICE_CLASSIFICATION_CONFLICT_PROJECTION
TOP3_NEXT=NONE_ACTIONABLE_FUNCTIONAL_ANCHOR_FAILED_AND_RETRY_PROHIBITED
SELECTED_NEXT_MECHANISM=FIXED_E1_CALIBRATED_RELIABILITY_RESIDUAL_STAGE_FUSION
WHY_SELECTED=It directly reduces the measured harmful late-stage contribution, leaves anomaly residuals unconstrained, has a favorable frozen source counterfactual, and is simpler than conflict projection or a learned gate.
WHY_NOT_CONFLICT_PROJECTION=Conflict evidence is only POSSIBLE, no severe conflict rule was met, complete per-term gradients are unavailable, and it is more invasive than the direct stage-contribution intervention.
WHY_NOT_STAGE_FUSION=NOT_APPLICABLE; stage fusion is selected. The protocol uses a fixed source-calibrated residual rather than a learned or swept gate.
NEXT_BOUNDED_EXPERIMENT_JUSTIFIED=YES
NEW_TRAINING_RUN=NO
OPTIMIZER_STEP_USED=NO
MEDICAL_INFERENCE_RUN=NO
MVTEC_INFERENCE_RUN=NO
FULL_E15_STARTED=NO
TARGET_TUNING_USED=NO
```
