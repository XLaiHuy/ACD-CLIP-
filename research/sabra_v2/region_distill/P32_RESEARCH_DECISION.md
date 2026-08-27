# P32 Research Decision — Downstream-Invariant Forensic and Mechanism Design

Status: `RESEARCH_DECISION_COMPLETE`

Protocol identifier: `P32`

This artifact records an offline research decision. It is not authorization for
a P32 scientific Stage 2 run. No optimizer step, scientific model forward,
teacher forward, CLIP/Phase2B forward, cache rebuild, or held-result tuning was
performed.

## 1. Entry condition and inherited result

The previous P31 control comparison had terminated before this work. Its
authoritative terminal state was `P31_CONTROL_ALREADY_EXISTED`, at commit
`2f577889363e4b0ab67fcbe721267b22c157fb55` on branch
`research/p29r1-fast-objective-forensic-v1`. The worktree was clean at entry.
The parent forensic status was `FORENSIC_COMPLETE`.

The P31/P30R1 delta convention is explicit throughout this document:

```text
delta = P31/native - P30R1
```

The inherited forensic result is:

```text
PRIMARY_MECHANISM   = TEACHER_DIRECTION_NOT_CAUSAL
SECONDARY_MECHANISM = SPARSE_SELECTIVE_CORRECTION
```

The ranked P30R1 forensic hypotheses were:

1. `TEACHER_DIRECTION_NOT_CAUSAL`
2. `SPARSE_SELECTIVE_CORRECTION`
3. `DO_NO_HARM_NATIVE_PRESERVATION`
4. `DIRECTION_METRIC_ILL_CONDITIONED_BY_ABSTENTION`
5. `TEACHER_SCALE_REWEIGHTING`

The native control already existed and was slightly better on the locked
candle endpoints:

| endpoint | P31/native | P30R1 | P31/native − P30R1 |
|---|---:|---:|---:|
| pAP | 0.514140305 | 0.511513734 | +0.002626571 |
| pAUROC | 0.980667144 | 0.980534709 | +0.000132435 |

P30 nevertheless had substantially better raw direction metrics than P30R1
while having much worse pAP (`0.736924` cosine and `0.144618` pAP versus
`-0.070148` and `0.511514`). This is the causal contrast that invalidates raw
teacher-vector direction as the default scientific target. P30R1 changes are
descriptively sparse (`0.056409` effective residual support) and strongly
anomaly-enriched (`291.876x` absolute score-delta mass over area), but native
still wins, so usefulness is not established.

## 2. Exact deployment equations

The repository deployment contract is Gaussian blur followed by aligned
bilinear resize and stage averaging. Let `r_s[g]` be the student 9×9 scalar
margin residual for stage `g`, and `r_t` the cached 9×9 teacher target. Define

```text
D(r) = mean_g Interpolate_37_to_518( GaussianBlur_7,sigma=1(
          Interpolate_9_to_37(r[g]) ) )
```

where a single 9×9 target is repeated over the three stages when required.
The symmetric two-class construction adds `-r/2` to the normal logit and
`+r/2` to the abnormal logit. Therefore the exact deployed margin satisfies

```text
m_native  = deployed_logit[abnormal] - deployed_logit[normal]
Delta_m_s = m_student - m_native = D(r_s)
Delta_m_t = m_teacher - m_native = D(r_t)
p_abnormal = sigmoid(m_native + Delta_m)
```

The cancellation is exact because the deployment map before softmax is linear
and the common two-class offset is absent. A fixed operator analysis on the
frozen geometry produced a 268,324×81 matrix with rank 81 and condition number
`6.14036`. Thus a shared 9×9 margin effect is identifiable; the proposed
effect target is not a hidden way to remove a true nullspace. It is a
deployment-aware reweighting of the stage-mean residual, and this limitation
is part of the preregistered novelty position.

## 3. Offline mechanism checks

The exact deterministic deployment path reproduced the stored P30R1 result:

- margin-effect error against the stored student residual on a deterministic
  prefix: `7.24e-6` maximum;
- probability-delta error against the frozen prediction maps: `2.384e-7`
  maximum and `6.85e-11` mean absolute error;
- the student stage-mean null component was numerical roundoff only;
  the replicated stage-mean and total residual L2 norms agreed to about
  `1e-7` relative scale.

For the descriptive frozen candle comparison, the effect-space teacher/student
transfer was still poor under the already-trained P30R1: overall cosine
`-0.118337`, Pearson `0.280857`, and sign agreement `0.166892`. Conditioning
on the inherited raw-vector abstention threshold (`0.773205`) left 92/200
images in the abstention group and 108/200 active; active-group effect cosine
was `0.477763`, while the abstention-group cosine was `-0.818107`. This does
not evaluate a P32 model: P30R1 was trained with the old objective. It does
show that P30R1 did not already transfer the proposed effect accidentally.

The exact fixed O(N) local-neighbor effect-order diagnostic agreed on only
`0.292524` of 33,540 sampled horizontal/vertical neighbor comparisons per
image (ties included). In contrast, the native/P30R1 anomaly-score ranking was
already highly similar: pooled Spearman `0.974894` and top-1% overlap
`0.956680`. Global ranking is therefore largely redundant with native in this
scope; a ranking objective would add complexity without an identified causal
gap.

## 4. Source-only actionability and scale

The unique Tier-B source-cache union contained 2,162 samples from 23,782
exposures, with zero duplicate-value mismatches. No held mask or held outcome
was used in this source pass. On source teacher targets:

- exact-zero sample fraction was `0.004163`;
- raw teacher RMS q01/q50/q99 was `0.196889 / 2.056865 / 4.960112`;
- downstream-effect RMS q01/q50/q99 was `0.160942 / 1.989157 / 4.960111`;
- the fixed 256-pixel probe mean-absolute effect had mean `2.570551`;
- inherited P30R1 teacher-scale inverse-weight q99/q01 was `24.5616`.

Source-only evidence can identify whether a cached teacher intervention is
zero, its support, and its approximate deployed-effect magnitude. It cannot
identify whether that intervention is useful on future held anomalies without
using held outcomes. Therefore a learned actionability gate would be
underdetermined here. No source-derived threshold or category-specific rule
is proposed.

## 5. Targeted literature findings

The search was limited to the mechanism actually supported by the forensic and
stopped after the candidate ranking stopped changing. The entries below use
the requested format: problem → mechanism → equation → relevance → limitation.

1. **Hinton, Vinyals & Dean (2015), Distilling the Knowledge in a Neural
   Network.** Problem: compress a large teacher/ensemble → mechanism: softened
   output distribution transfer → equation:
   `T² KL(softmax(z_t/T) || softmax(z_s/T))` → relevance: establishes learned
   input/output behavior as a legitimate distillation target → limitation:
   transfers the full teacher output and does not isolate a native-relative
   anomaly intervention. [Primary paper](https://arxiv.org/abs/1503.02531)

2. **Penso, Achituve & Fetaya (2022), Functional Ensemble Distillation.**
   Problem: ordinary ensemble outputs omit useful predictive uncertainty and
   covariance → mechanism: distill the predictive function with augmentation
   to expose functional variation → equation: predictive-distribution/function
   matching with mixup-augmented inputs → relevance: closest “function rather
   than parameter/vector” precedent → limitation: Bayesian ensemble setting;
   it does not define a native-relative dense margin effect or anomaly
   residual intervention. [Primary paper](https://arxiv.org/abs/2206.02183)

3. **Zheng et al. (2022), Localization Distillation for Object Detection.**
   Problem: feature imitation can omit localization knowledge → mechanism:
   distill teacher prediction distributions over valuable localization regions
   → equation: region-weighted KL between teacher and student localization
   distributions → relevance: direct evidence that task-output/logit
   behavior can be more useful than internal feature imitation, with no
   inference-speed sacrifice reported → limitation: detection-specific heads,
   boxes, and region selection; not anomaly residuals. [Primary paper](https://arxiv.org/abs/2204.05957)

4. **Park et al. (2019), Relational Knowledge Distillation.** Problem:
   pointwise matching misses relations between examples → mechanism: match
   normalized pairwise distances and angles → equation:
   `lambda_D L_distance + lambda_A L_angle` → relevance: established
   structure-preserving alternative to raw vector matching → limitation:
   pairwise structure and multiple weighted terms; no dense anomaly-logit
   intervention. [Primary paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html)

5. **Reddi et al. (2021), RankDistil: Knowledge Distillation for Ranking.**
   Problem: top-k ordering is the target in large ranking systems → mechanism:
   teacher-ordered top-k/listwise or pairwise losses with negative sampling →
   equation: pairwise form `Psi(t,s,P) + sum_{i in N,j in P} phi(s_j-s_i)` →
   relevance: primary precedent for rank-preserving transfer and efficient
   sampling → limitation: large-item ranking, extra pair conventions, and no
   image-anomaly or residual-effect validation. [Primary paper](https://proceedings.mlr.press/v130/reddi21a.html)

6. **Bergmann et al. (2020), Uninformed Students.** Problem: unsupervised
   pixel anomaly detection → mechanism: regress pretrained teacher features on
   normal data and score the discrepancy → equation:
   `A(x)=d(f_teacher(x), f_student(x))` → relevance: direct teacher–student
   anomaly-detection precedent → limitation: the teacher/student discrepancy
   remains an inference-time signal and is not a native-relative output
   effect. [Primary paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Bergmann_Uninformed_Students_Student-Teacher_Anomaly_Detection_With_Discriminative_Latent_Embeddings_CVPR_2020_paper.html)

7. **Wang et al. (2021), Student-Teacher Feature Pyramid Matching for Anomaly
   Detection.** Problem: one-class anomaly localization across scales →
   mechanism: multi-scale teacher/student feature matching → equation:
   multi-level feature discrepancy used as the anomaly map → relevance:
   confirms that anomaly detection usually scores downstream discrepancy, not
   residual direction → limitation: teacher-feature discrepancy and a
   different inference architecture; no native-relative margin effect.
   [Primary paper](https://arxiv.org/abs/2103.04257)

8. **Geifman & El-Yaniv (2019), SelectiveNet.** Problem: prediction under a
   reject/abstain option → mechanism: learn prediction and rejection jointly
   under a coverage constraint → equation:
   `coverage=E[g(x)]`, `risk=E[l(f(x),y)g(x)]/coverage` → relevance: formalizes
   why abstention/actionability needs a declared coverage or utility criterion
   → limitation: adds a learned gate and labeled selective-risk objective;
   that is outside the zero-new-module SABRA scope. [Primary paper](https://proceedings.mlr.press/v97/geifman19a.html)

The reviewed set establishes prior art for output/function, relational, rank,
selective, and teacher–student anomaly transfer. No exact native-relative
SABRA region-residual margin intervention was found in this targeted set. This
is a scoped overlap statement, not a novelty claim.

### Exact overlap audit

| question | conclusion |
|---|---|
| exact mechanism already exists? | The components exist separately; the exact fixed native-relative dense margin intervention was not found in the targeted set. |
| anomaly detection? | Yes: Uninformed Students and STFPM; they use feature discrepancy rather than this effect. |
| teacher–student anomaly detection? | Yes, but no reviewed paper used this exact cached native-relative residual effect. |
| residual/logit intervention? | Logit/output transfer is established in classification/detection; the anomaly residual-to-native-margin combination remains unverified. |
| extra inference cost? | Candidate A adds no teacher or extra branch at inference; SelectiveNet/MemKD-like gates or memories would. |
| multiple weighted losses? | RKD, RankDistil variants, selective models, and task-aware detection methods often add relation/pair/coverage machinery. Candidate A uses one loss. |
| category-specific parameters? | Not required by the reviewed mechanisms; none are proposed for P32. |

## 6. Candidates

### Candidate A — `FUNCTIONAL_MARGIN_EFFECT`

- **Mechanism:** match the deployed native-relative margin effect, not the raw
  243-coordinate residual direction:
  `L_FME = mean SmoothL1(D(mean_stage(r_s)), stopgrad(D(r_t)), beta=1.0)`.
- **Forensic support:** directly answers `TEACHER_DIRECTION_NOT_CAUSAL`; the
  exact deployment equations identify the effect; P30/P30R1 show raw direction
  and detection can decouple.
- **Scientific question:** can a student transfer teacher-induced downstream
  margin changes while remaining safe relative to the native detector?
- **Expected detection effect:** preserve any useful teacher-induced anomaly
  score changes without requiring raw residual direction; no improvement over
  native is assumed.
- **Expected residual behavior:** stage-mean residual is constrained through
  the deployment operator; stage-specific components are not separately
  constrained.
- **Teacher fidelity:** matters only in `Delta_m`, never raw vector cosine/sign.
- **Objectives:** 1. **New hyperparameters:** 0; beta `1.0` is inherited.
- **Cost:** training-only fixed blur/resize on one stage-mean map; target
  end-to-end overhead ≤10% preferred and ≤15% hard; inference overhead 0%.
- **Novelty risk:** medium/partial prior-art overlap; no method novelty claim.
- **Main failure mode:** the teacher effect is itself unhelpful or the linear
  pullback merely changes conditioning without downstream benefit.
- **Cheapest falsification:** symbolic full-rank/linearity check, adversarial
  finite-gradient suite, source target audit, then one future locked comparison
  against native; failure of any declared outcome/safety gate stops the method.

### Candidate B — `LOCAL_RANKING_EFFECT`

- **Mechanism:** preserve teacher ordering of local neighboring deployed
  effects using a fixed O(N) neighbor set, not all-pairs ranking.
- **Forensic support:** ranking is downstream-facing in principle, but the
  frozen native/P30R1 global rank is already highly redundant and local
  effect-order agreement is only `0.292524`.
- **Scientific question:** is relative local ordering, rather than effect
  magnitude, the useful teacher invariant?
- **Expected detection effect:** possible topological/ranking preservation but
  no calibrated effect scale or safety guarantee.
- **Expected residual behavior:** many materially different residuals satisfy
  the same order constraints.
- **Teacher fidelity:** only teacher local ordering.
- **Objectives:** 1, but pair/tie conventions are new design choices; train
  cost and sensitivity are higher than A; inference 0%.
- **Novelty risk:** low-medium because rank distillation is established.
- **Main failure mode:** metric chasing and recurrence of the old
  overconstraint pattern; global rank is already native-like.
- **Cheapest falsification:** fixed finite non-tied pair inventory and the
  frozen native redundancy audit; no held-based pair selection.

### Candidate C — `SELECTIVE_ACTIONABILITY_ABSTENTION`

- **Mechanism:** transfer only source-defined nonzero/high-actionability
  teacher corrections, without learning a gate; otherwise preserve native.
- **Forensic support:** P30R1 correction is sparse and anomaly-enriched, and
  source cache contains measurable zero/support/effect distributions.
- **Scientific question:** can a fixed source-only actionability rule prevent
  harmful global intervention?
- **Expected detection effect:** lower native damage if weak corrections are
  omitted, but no source-only proof of held utility.
- **Expected residual behavior:** more abstention/less support and exact native
  preservation outside selected actions.
- **Teacher fidelity:** only for selected actionability, not direction.
- **Objectives:** 0 if evaluation-only; a learned gate would be an additional
  module and is rejected. Any threshold would need an inherited or analytic
  definition; no held tuning is allowed.
- **Cost:** 0% for the diagnostic; a rule at training/inference adds branch
  complexity; no teacher at inference is allowed.
- **Novelty risk:** low; selective prediction and confidence filtering are
  established.
- **Main failure mode:** actionability is not identifiable from source-only
  scale/support, so the rule becomes an unregistered threshold.
- **Cheapest falsification:** show that all available source-only rules reduce
  to magnitude selection without an outcome-free, preregisterable utility
  criterion. This is the observed state; no gate is proposed.

## 7. Candidate comparison

| Candidate | Mechanism match | Objectives | Hyperparams | Train cost | Inference cost | Overconstraint risk | Novelty | Falsifiability |
|---|---|---:|---:|---:|---:|---|---|---|
| A: functional margin effect | high/direct | 1 | 0 new | fixed one-map transform; ≤10% target | 0% | medium-low | medium/partial | high |
| B: local ranking effect | medium | 1 | pair/tie choices | higher/variable | 0% | high | low-medium | medium |
| C: selective actionability | partial/underdetermined | 0 diagnostic or gate | threshold required | 0% diagnostic | 0% if no gate | medium-high if learned | low | high as a stop |

The ranking is by mechanism validity × simplicity × falsifiability ×
performance potential × runtime × publishability. Candidate C is retained as
an analysis/control principle, not as a learned gate. Candidate B is rejected
as the primary experiment because global ordering is already native-like and
the local diagnostic is weak. Candidate A is the only candidate that tests the
declared causal question without adding a second scientific rationale.

## 8. SELECTED_P32_HYPOTHESIS

`SELECTED_P32_HYPOTHESIS = FUNCTIONAL_MARGIN_EFFECT`

P32 tests the causal claim:

> If teacher direction is not causal, then matching the teacher's exact
> native-relative deployed margin effect can transfer downstream-relevant
> intervention without requiring raw residual-vector direction, while a
> native/zero-adapter control remains the safety reference.

This is motivated by the P29/P30/P30R1 sequence: P29 mixed objectives, P30
made direction explicit but lost radial identifiability and detection, and
P30R1 restored radial regression and detection while raw direction collapsed.
The P31 native control then showed that P30R1 had not demonstrated benefit over
native. A P32 result may therefore legitimately conclude that native is best;
effect transfer is a falsifiable test, not an assumption that teacher
imitation must remain in SABRA.

The simplest exact formulation is one robust SmoothL1 objective on the
deployment operator's native-relative margin effect. It has no cosine, sign,
Pearson, Spearman, ranking, normal-pixel, sparsity, gate, or auxiliary term.
The fixed operator is full-rank, so the experiment tests a deployment-aware
conditioning/target—not a claim that residual radial scale is mathematically
unidentifiable. It does not repeat the old overconstraint pattern because it
has one mechanism, one objective, zero new tuned scalars, no category-specific
parameters, and no inference-time teacher.

## 9. Cheapest falsification plan

The ordered gates are:

1. symbolic check of the exact repository deployment equations and fixed
   operator rank/conditioning;
2. deterministic zero, near-zero, scale-mismatch, sign-reversal, sparse,
   heavy-tail, mixed-scale, outlier, all-null, and high-effect gradient tests;
3. source-cache-only scale/support/effect distribution and category-dispersion
   audit, with no held labels;
4. frozen descriptive check that the old P30R1 result did not already satisfy
   the proposed effect target;
5. only after a separate implementation freeze, one cached-batch engineering
   smoke and profile;
6. only after explicit future authorization, one locked scientific Stage 2
   comparison with native/zero-adapter and P30R1 frozen references.

The exact scientific falsifier is: after prediction freeze, the candidate fails
the predeclared primary pAP or secondary pAUROC non-inferiority comparison to
the frozen native control, or fails any finite-gradient, normal-preservation,
residual-tail, provenance, or runtime gate. A failure stops P32 and does not
authorize a ranking/gating rescue.

## 10. Runtime, data, and execution boundary

The future candidate uses `[3,B,9,9]` student residuals and `[B,9,9]` cached
teacher targets. Its training objective is O(BHW) after one vectorized
stage-mean deployment transform; there is no pairwise O(N²) operation, no
extra network, no teacher forward, and no inference branch. The native control
is always retained.

This research phase used no optimizer steps, no new CLIP or Phase2B forwards,
no cache rebuild, no Stage 2/Stage 3/full run, and no scientific execution
marker. The 100 held mask reads in the effect reconstruction were post-freeze,
descriptive forensic reads inherited from the P30R1 causal analysis; they were
not used to tune a P32 parameter or threshold.

The preregistration draft for the selected formulation is
`research/sabra_v2/region_distill/P32_PREREGISTRATION_DRAFT.md`. It remains a
draft until the formulation/preflight implementation work is frozen under a
separate explicit engineering step.

`P32_RESEARCH_DECISION_COMPLETE`
