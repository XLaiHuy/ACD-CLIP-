# P31 Research Decision — Native Control After P30R1 Causal Forensic

Status: `RESEARCH_DECISION_COMPLETE`

Protocol identifier: `P31`

This is a research decision and preregistration-draft handoff only. It is not
execution authorization. No model, checkpoint, trainer, runner, scientific
marker, or existing scientific artifact was modified.

## 1. Entry condition

The required terminal artifacts were present before this decision was made:

- `P30R1_FORENSIC/P30R1_CAUSAL_FORENSIC_REPORT.md`
- `P30R1_FORENSIC/P30R1_CAUSAL_FORENSIC.json`

The machine-readable forensic status is `FORENSIC_COMPLETE`. The forensic
commit is `0800cf86d6896b164885768a0b6653a8f1953a76` on branch
`research/p29r1-fast-objective-forensic-v1`; the worktree was clean at entry.
The ambiguous/data-access stop condition was therefore not triggered.

## 2. Forensic result summary

The authoritative primary mechanism is:

`PRIMARY_MECHANISM = TEACHER_DIRECTION_NOT_CAUSAL`

The secondary mechanism is:

`SECONDARY_MECHANISM = SPARSE_SELECTIVE_CORRECTION`

The ranked forensic hypotheses were:

1. `TEACHER_DIRECTION_NOT_CAUSAL`
2. `SPARSE_SELECTIVE_CORRECTION`
3. `DO_NO_HARM_NATIVE_PRESERVATION`
4. `DIRECTION_METRIC_ILL_CONDITIONED_BY_ABSTENTION`
5. `TEACHER_SCALE_REWEIGHTING`

The decision-relevant frozen evidence is:

- P30 had much higher teacher-direction fidelity than P30R1 but much worse
  pAP: approximately `0.7369` cosine / `0.1446` pAP versus P30R1's
  approximately `-0.0701` cosine / `0.5115` pAP. Direction is therefore not
  a validated downstream proxy.
- P30R1 is close to the native detector on most pixels, with per-image
  Pearson approximately `0.9478`, top-1% overlap approximately `0.9567`, and
  small global mean absolute score change. Its correction is sparse and
  anomaly-enriched, but this is not proof that the correction is useful.
- The exact frozen native/zero-adapter counterfactual has pAP approximately
  `0.514140` and pAUROC approximately `0.980667`; P30R1 has pAP approximately
  `0.511514` and pAUROC approximately `0.980535`. Native is slightly better
  on both reported outcomes.
- Low student norm explains only part of the directional collapse: the
  highest-norm P30R1 bin has cosine approximately `0.6220` but still poor sign
  agreement. Abstention is not a sufficient rescue for the direction metric.
- P30R1's teacher-only scale denominator has an inverse-weight q99/q01 spread
  of approximately `24.56`. This makes scale reweighting plausible, but does
  not establish downstream causality.

The frozen forensic question is therefore routed to Route C:

> What downstream-relevant effect, if any, should be transferred instead of
> raw teacher correction direction?

The first answer must be whether any transfer is needed. Because the native
control is already comparable or better, a zero-objective native control is
scientifically stronger than immediately inventing a new distillation loss.

## 3. Targeted Route C literature findings

This was a targeted eight-paper review, limited to output/function,
decision/rank, task-aware, and teacher–student anomaly-distillation mechanisms.
The search stopped when additional results repeated these mechanisms. Equations
below are compact descriptions of the cited papers' transfer targets, not
claims that their settings are identical to SABRA.

### 3.1 Useful papers

1. **Hinton, Vinyals, and Dean (2015), “Distilling the Knowledge in a Neural Network.”**
   Problem: compress an ensemble or large teacher. Mechanism: match softened
   teacher outputs, `L_KD = T² KL(softmax(z_T/T) || softmax(z_S/T))`, or its
   high-temperature logit approximation. Relevance: establishes output/function
   transfer as an alternative to internal-vector imitation. Limitation: it
   transfers the teacher's complete output behavior and does not identify a
   useful anomaly residual or intervention effect. [Primary paper](https://arxiv.org/abs/1503.02531)

2. **Park et al. (2019), “Relational Knowledge Distillation.”** Problem:
   pointwise outputs can omit structure between examples. Mechanism: match
   normalized pairwise distances and angles, approximately
   `L_RKD = λ_D L_distance + λ_A L_angle`. Relevance: shows that a downstream
   invariant need not be a per-example vector direction. Limitation: it adds
   pairwise structure and two relation terms, was evaluated for representation
   and metric learning, and does not establish anomaly-score or residual-logit
   causality. [Primary paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html)

3. **Chen, Wang, and Zhang (2018), “DarkRank.”** Problem: transfer metric
   structure rather than only individual embeddings. Mechanism: a learning-to-
   rank loss preserves teacher cross-sample similarity ordering, schematically
   `L_rank = −Σ log exp(s_S(i,j)) / Σ_k exp(s_S(i,k))` under teacher-derived
   orderings. Relevance: direct prior art for rank-preserving transfer.
   Limitation: it is a metric-learning method with pair/list construction and
   no teacher–student anomaly or residual-logit intervention result. [Primary paper](https://ojs.aaai.org/index.php/AAAI/article/view/11783)

4. **Zheng et al. (2022), “Localization Distillation for Dense Object
   Detection.”** Problem: feature imitation can miss localization information.
   Mechanism: distill prediction-head localization distributions with a
   teacher/student divergence, `L_LD = mean_i w_i KL(p_T,i || p_S,i)`, over
   valuable localization regions; classification and localization logits are
   treated as task outputs. Relevance: strong evidence that downstream
   prediction/logit effects can be more relevant than feature imitation, with
   no reported inference-speed sacrifice. Limitation: object-detection boxes,
   region heuristics, and a classification-plus-localization formulation do
   not transfer exactly to SABRA's one-class residual adapter. [Primary paper](https://arxiv.org/abs/2204.05957)

5. **Wang et al. (2024), “CrossKD: Cross-Head Knowledge Distillation for
   Object Detection.”** Problem: feature imitation can conflict with annotation
   supervision. Mechanism: pass student head features through the teacher head
   and match predictions, `L_CrossKD = mean_r S(r) D_pred(p̂_S(r), p_T(r))`.
   Relevance: the closest general precedent for task-oriented prediction
   transfer and for separating the distillation prediction from conflicting
   internal supervision. Limitation: it uses a teacher detection head and
   detection-specific prediction losses; it is not a frozen anomaly residual
   intervention and the target teacher prediction may still be wrong. [Primary paper](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.html)

6. **Lan and Tian (2024), “Gradient-Guided Knowledge Distillation for Object
   Detectors.”** Problem: plain feature imitation does not identify features
   that affect the task. Mechanism: weight feature imitation by task-loss
   gradients, approximately `L_GKD = mean_i |∂L_task/∂f_i| D(f_S,i, f_T,i)`,
   with an additional bounding-box-aware multi-grained imitation component.
   Relevance: demonstrates a downstream-aware weighting mechanism. Limitation:
   it requires task-gradient computation and extra feature machinery, uses
   object-detection supervision, and does not show anomaly residual/logit
   transfer. [Primary paper](https://openaccess.thecvf.com/content/WACV2024/html/Lan_Gradient-Guided_Knowledge_Distillation_for_Object_Detectors_WACV_2024_paper.html)

7. **Bergmann et al. (2020), “Uninformed Students.”** Problem: unsupervised
   pixel anomaly detection without anomaly labels. Mechanism: students regress
   frozen teacher features on normal data and score discrepancy,
   `A(x) = d(f_T(x), f_S(x))`, with ensemble uncertainty as an additional
   signal. Relevance: direct teacher–student anomaly-detection precedent.
   Limitation: the teacher/student feature discrepancy is the anomaly score,
   so teacher computation remains part of inference; it does not replace raw
   feature transfer with a frozen native-relative logit effect. [Primary paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Bergmann_Uninformed_Students_Student-Teacher_Anomaly_Detection_With_Discriminative_Latent_Embeddings_CVPR_2020_paper.html)

8. **Gu et al. (2023), “Remembering Normality.”** Problem: student–teacher AD
   can forget normality and either reconstruct anomalies or overreact to fine
   normal patterns. Mechanism: a normality-recall memory modulates student
   features before discrepancy scoring, with a normality-embedding objective.
   Relevance: confirms that AD distillation literature targets the behavior of
   the downstream anomaly score, not teacher direction alone. Limitation: it
   adds a memory module and an auxiliary normality-learning path, and remains
   feature-discrepancy based rather than a native-relative logit intervention.
   [Primary paper](https://openaccess.thecvf.com/content/ICCV2023/html/Gu_Remembering_Normality_Memory-guided_Knowledge_Distillation_for_Unsupervised_Anomaly_Detection_ICCV_2023_paper.html)

### 3.2 Exact prior-art overlap

| Question | Finding from the targeted set |
|---|---|
| Does the exact mechanism already exist? | Output/logit distillation, relational transfer, rank transfer, task-gradient weighting, and teacher–student AD each already exist. No exact match to frozen native-relative SABRA region-residual intervention was identified in this targeted set. This is not a novelty claim. |
| Has it been applied to anomaly detection? | Yes for teacher–student feature discrepancy and normality memory (Uninformed Students; MemKD). |
| Has it been applied to teacher–student anomaly detection? | Yes, but the reviewed AD methods use teacher/student feature discrepancy or memory modulation, not the proposed native-relative output effect. |
| Has it been applied to residual/logit intervention? | Prediction/logit transfer is established in classification and object detection (KD, LD, CrossKD). A direct anomaly residual-to-native-logit intervention was not found in the targeted set. |
| Does it require extra inference cost? | Standard KD, LD, and CrossKD can discard the teacher/distillation branch at deployment; LD explicitly reports no inference-speed sacrifice. Uninformed Students retains teacher/student discrepancy at inference, and MemKD adds an inference memory path. |
| Does it require multiple weighted losses? | RKD uses distance and angle terms; GKD adds BMFI; MemKD adds normality-memory learning. CrossKD/LD are task-specific prediction objectives used alongside the detector's ordinary supervised objective. A null control requires zero objectives. |
| Does it introduce category-specific parameters? | The reviewed formulations are generally shared across categories; their complexity comes from relation pairs, regions, task heads, gradients, or memory, not category-specific SABRA parameters. |

The literature supports investigating downstream effects in principle. It does
not override the frozen native counterfactual, and it does not justify adding a
ranking or auxiliary loss merely to repair P30R1's direction/sign diagnostics.

## 4. Anti-overengineering filter

The following are rejected as primary next steps at this point:

- repairing cosine, sign, Pearson, or Spearman directly;
- adding a ranking loss because a direction metric failed;
- adding L1 sparsity because the correction is sparse descriptively;
- adding a learned gate, auxiliary network, teacher-at-inference path, or
  category-specific tuning;
- tuning an intervention threshold, coefficient, or loss weight on the held
  evidence.

The forensic already shows a near-native result, while native is slightly
better. The zero-adapter branch therefore has the highest mechanism validity,
lowest overconstraint risk, and cheapest falsification cost.

## 5. Candidate hypotheses

### Candidate 1 — `P31_NATIVE_ZERO_ADAPTER_CONTROL`

- **Exact mechanism:** set the learned residual to zero: `r_S(x) = 0`, so the
  deployed logits and anomaly map equal the frozen native outputs.
- **Forensic support:** native is at least as good as P30R1 on the reported
  pAP/pAUROC pair, while P30R1's close-to-native behavior suggests damage
  avoidance rather than a demonstrated teacher correction gain.
- **Scientific question:** does teacher intervention provide any downstream
  benefit over the frozen detector on the locked one-class comparison?
- **Expected detection effect:** no adapter-induced improvement; no adapter-
  induced degradation. Native wins if the teacher correction is unnecessary.
- **Expected residual behavior:** exactly zero; score delta from native exactly
  zero.
- **Does teacher fidelity matter?** No. There is no teacher target.
- **Objectives:** `0`.
- **New hyperparameters:** `0`.
- **Training overhead:** `0%`.
- **Inference overhead:** `0%`; use the existing native output.
- **Novelty risk:** no method novelty is claimed; this is a necessary control.
- **Main failure mode:** a teacher correction may recover a systematic anomaly
  effect that native misses on another locked class.
- **Cheapest falsification:** compare cached native and P30R1 held predictions
  with the locked pAP/pAUROC comparison; no forward or optimizer step.

### Candidate 2 — `DOWNSTREAM_LOGIT_EFFECT_TRANSFER`

- **Exact mechanism:** transfer the teacher's native-relative deployed-logit
  effect rather than its internal residual direction. With native logits
  `ℓ_0`, frozen teacher-effect logits `ℓ_T`, and student-effect logits `ℓ_S`,
  use one target `Δℓ_T = ℓ_T − ℓ_0`, `Δℓ_S = ℓ_S − ℓ_0`, and one objective
  `L_effect = mean SmoothL1(Δℓ_S, stopgrad(Δℓ_T); beta=1)`.
- **Forensic support:** directly answers the recommended question and allows
  the student residual direction to differ when only the downstream logit
  effect is useful.
- **Scientific question:** can a frozen native-relative output effect transfer
  useful teacher behavior without reimposing raw residual-direction fidelity?
- **Expected detection effect:** preserve only teacher-induced changes that
  survive the deployment logit operator; possibly retain sparse anomaly-local
  corrections.
- **Expected residual behavior:** residual direction may differ from the
  teacher; native-relative logit effect should match the frozen target.
- **Does teacher fidelity matter?** Only at the deployed-logit effect, not in
  internal direction or sign.
- **Objectives:** `1`.
- **New hyperparameters:** `0` in the candidate formulation; existing SmoothL1
  beta is inherited and not tuned.
- **Training overhead:** expected low single-path overhead, target `<10%`, but
  must be measured before authorization; no extra neural teacher forward if
  the target is cached.
- **Inference overhead:** `0%` if the target is training-only and the existing
  adapter deployment path is retained.
- **Novelty risk:** medium; prediction/logit distillation is established, so
  any claim would have to be limited to this native-relative anomaly setting.
- **Main failure mode:** the teacher's output effect may itself be useless or
  may collapse to the native output; matching it could preserve the wrong
  intervention.
- **Cheapest falsification:** verify from frozen source artifacts that
  `Δℓ_T` is finite, nonzero, and identifiable without held-label selection; a
  zero/unstable target kills the candidate before training.

### Candidate 3 — `DOWNSTREAM_RANK_OR_MARGIN_TRANSFER`

- **Exact mechanism:** preserve teacher ordering/margins of native-relative
  effects, for example `L_rank = mean softplus(−sign(Δℓ_T,i − Δℓ_T,j)
  (Δℓ_S,i − Δℓ_S,j))` over non-tied pairs.
- **Forensic support:** ranking is downstream-adjacent and avoids direct
  direction matching, but the forensic did not establish ranking as the
  causal invariant; this is a lower-ranked exploratory alternative.
- **Scientific question:** is anomaly ordering, rather than effect magnitude,
  the useful teacher signal?
- **Expected detection effect:** possible ranking/pAP improvement with no
  guarantee of calibrated score magnitude.
- **Expected residual behavior:** many residuals can differ while pairwise
  ordering is preserved.
- **Does teacher fidelity matter?** Only for teacher-derived ordering; raw
  vector direction does not matter.
- **Objectives:** `1`, but it requires a pair construction rule.
- **New hyperparameters:** at least a tie rule and pair sampling/temperature
  convention unless all non-tied pairs are used; these must not be held-label
  tuned.
- **Training overhead:** potentially high because of pair/list construction;
  it is not compatible with the preferred `<10%` training-overhead target
  without a preflight measurement.
- **Inference overhead:** `0%`.
- **Novelty risk:** low-to-medium because rank transfer is established by
  DarkRank and related methods; exact anomaly residual use is unestablished.
- **Main failure mode:** metric chasing and unstable pair weighting can repeat
  the P29 overconstraint pattern while leaving absolute score behavior unsafe.
- **Cheapest falsification:** count finite, non-tied teacher-effect pairs in
  frozen source data; a mostly tied/near-zero target supplies no usable rank
  signal.

## 6. Candidate comparison and ranking

The ranking uses mechanism validity first, then simplicity, falsifiability,
performance potential, runtime, and publishability. Novelty is not allowed to
rescue a weak mechanism match.

| Candidate | Mechanism match | Objectives | Hyperparams | Train cost | Inference cost | Overconstraint risk | Novelty | Falsifiability |
|---|---:|---:|---:|---:|---:|---|---|---|
| `P31_NATIVE_ZERO_ADAPTER_CONTROL` | highest | 0 | 0 | 0% | 0% | low | control/no claim | immediate, cached |
| `DOWNSTREAM_LOGIT_EFFECT_TRANSFER` | high but unvalidated | 1 | 0 new | target <10% | 0% | medium | medium / partial prior art | frozen target audit |
| `DOWNSTREAM_RANK_OR_MARGIN_TRANSFER` | medium | 1 | ≥2 conventions | potentially high | 0% | high | low–medium / known family | pair-signal audit |

## 7. Selected next hypothesis

`SELECTED_NEXT_HYPOTHESIS = P31_NATIVE_ZERO_ADAPTER_CONTROL`

The causal claim is:

> On the locked SABRA comparison, the frozen native detector is non-inferior
> to the P30R1 teacher-residual intervention; therefore raw teacher imitation
> is not necessary as a default component of SABRA unless a separately
> preregistered future test demonstrates a downstream gain.

Why this is selected:

1. P29/P29R1 showed the cost of mixed objectives and competing signals.
   P30 showed that high directional fidelity can coexist with catastrophic
   downstream behavior. P30R1 restored radial control and detection behavior
   while direction collapsed. Together they make raw direction an unsupported
   target, not a missing metric to repair.
2. The exact native counterfactual is slightly better than P30R1 on both
   reported outcomes. This is the strongest available evidence against
   inventing a learned method merely to continue the protocol sequence.
3. The control directly tests the causal necessity of teacher intervention,
   introduces zero new loss terms and zero new parameters, and has no
   inference cost.
4. The literature provides plausible future output-effect mechanisms, but
   their settings do not establish that the P30R1 teacher effect is useful for
   this frozen one-class detector. Candidate 2 remains a contingent follow-up,
   not an authorization.

The selected hypothesis is falsified if the locked native control is worse than
P30R1 on the predeclared primary comparison. That would reject the control for
this scope; it would not authorize ranking loss, a gate, or a new training run.

## 8. Cheapest falsification plan

1. Validate the existing forensic JSON/report status, input identities, tensor
   shapes, and native reconstruction tolerance.
2. Compare the already cached native zero-adapter and P30R1 held predictions
   with the inherited deterministic pAP primary metric and pAUROC secondary
   metric; use zero non-inferiority margin and no threshold or coefficient
   tuning.
3. If native is at least as good on both metrics, accept the null control and
   stop teacher-imitation research for this scope. If P30R1 wins either locked
   comparison, mark the control falsified and stop for a new research decision;
   do not automatically implement Candidate 2 or 3.

This sequence uses cached outputs only. It requires no optimizer step, no new
CLIP/Phase2B/teacher forward, no cache rebuild, and no Stage 2/Stage 3/full
run.

## 9. Overconstraint risk assessment

Selected control: `LOW`.

It has no learnable intervention, no loss weights, no gate, no auxiliary
network, no category-specific parameter, and no teacher at inference. The
rejected learned candidates are `MEDIUM` and `HIGH` risk respectively because
they could turn a downstream question into another overconstrained imitation
objective.

## 10. Runtime, inference, and training decision

| Item | P31 selected control |
|---|---:|
| New training runs | `0` |
| Optimizer steps | `0` |
| New CLIP forwards | `0` |
| New Phase2B forwards | `0` |
| New teacher forwards | `0` |
| Stage 2 runs | `0` |
| Stage 3 runs | `0` |
| Full 12-class runs | `0` |
| New objectives | `0` |
| New hyperparameters | `0` |
| Training overhead | `0%` |
| Inference overhead | `0%` |

`TRAINING_REQUIRED = NO`.

No new scientific execution is required to make this decision. If a future
learned Candidate 2 is considered, it requires a separate preregistration and
new engineering qualification after the native-control result; this artifact
does not authorize it.

## 11. Handoff artifacts

- Decision: `research/sabra_v2/region_distill/P31_RESEARCH_DECISION.md`
- Machine-readable decision: `research/sabra_v2/region_distill/P31_RESEARCH_DECISION.json`
- Draft only: `research/sabra_v2/region_distill/P31_PREREGISTRATION_DRAFT.md`

No scientific UUID, execution marker, final preregistration hash, trainer,
runner, checkpoint, or model implementation was created.

`P31_RESEARCH_DECISION_COMPLETE`
