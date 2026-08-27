# P35 Research Decision — Soft Actionability Reweighting

Status: `P35_RESEARCH_DECISION_COMPLETE`

Protocol identifier: `P35`. This is an offline research and engineering
decision. It creates no P35 scientific UUID and authorizes no P35 Stage 2
run.

## 1. Entry condition and frozen P34 result

The authoritative P34 evidence is the synchronized commit
`5544ef7ba3984b1a3e4f750a32458a07e784b1c4`. P34 completed training,
prediction freezing, and post-freeze scoring, then stopped with a reporting
wrapper schema error. The frozen scientific result before that wrapper error
was `P34_STAGE2_SCIENTIFIC_STOP`.

| method | pAP | pAUROC |
|---|---:|---:|
| P31/native | 0.514140304931 | 0.980667143514 |
| P30R1 | 0.511513734224 | 0.980534708954 |
| P32 | 0.510351502947 | 0.971460700418 |
| P33 | 0.519395095936 | 0.978184288830 |
| P34 | 0.505450138378 | 0.970397710904 |

P34 was safe radially (`residual q99=4.913081169128`, normal-score q99
shift `1.629600774322e-06`) but lost detection quality. Its meaningful
intervention fraction was `0.875988`, effective support `0.495238`, Gini
`0.523965`, and top-10% residual mass `0.218835`, all less dense than P33.
Thus “more sparse” is not itself the mechanism of success.

P34's reporting defect is preserved and fixed only in engineering code. The
root cause and frozen-metadata regression are recorded in
[`P34_REPORTING_BUG_ROOT_CAUSE.md`](P34_REPORTING_BUG_ROOT_CAUSE.md). No P34
evidence was rewritten or rerun.

## 2. P34 forensic hypotheses and ranking

The five required hypotheses are ranked using the frozen P33/P34 endpoints,
selectivity diagnostics, and the source-only P34 cache audit. Ranking is not a
claim that a held label has identified the cause; the alternatives remain
explicitly falsifiable.

| rank | hypothesis | frozen support | cheap falsifier |
|---:|---|---|---|
| 1 | `DENSITY_NOT_PRIMARY` | P33 had the best pAP despite effective support `0.962760`; P34 reduced support but collapsed pAP/pAUROC | a source/synthetic gradient-allocation test showing P33's only material difference is output sparsity rather than optimization allocation |
| 2 | `SAMPLE_IMPORTANCE_IS_CAUSAL` | P33 retained the full functional target and improved pAP over native/P32; P34 changed the target and degraded both endpoints | a source/synthetic or future preregistered test in which preserving `E_t` while changing only importance cannot retain the P33 benefit |
| 3 | `TARGET_ATTENUATION_DESTROYS_USEFUL_ACTION` | P34 replaced `E_t` by `wE_t`, reduced source target magnitude, and was worse than P33 | an isolated target-preserving comparison that reproduces the P34 failure without target attenuation |
| 4 | `HARD_WEIGHT_SATURATION_LIMITS_P33` | inherited clamp gives weight 1 to `43.871699%` of source pixels; raw `x` has no mass at `x>=2` | source-only/synthetic analysis showing the clamp plateau does not change optimization ordering or that a soft map is numerically unstable |
| 5 | `ACTIONABILITY_SIGNAL_IS_INSUFFICIENT` | P33 still missed native pAUROC, so the proxy may be incidental | a source-only descriptor audit finding no stable, category-agnostic effect ordering, or a clean future full-target weighted test failing its locked endpoints |

The selected P35 test isolates ranks 2–4 without reusing P34 target shaping:
it preserves the full target and changes only the source-example importance map.

## 3. Exact source-only forensic

The analysis used only
`/workspace/p27r1_cache_v1/tier_b/candle/teacher_region.npy` and its source
manifest. There were `1,962` source records and `526,451,688` deployed-effect
pixels. The reproducible implementation is
[`p35_soft_actionability.py`](../../../tools/sabra_v2/forensics/p35_soft_actionability.py).
The systematic sample contains `299,973` pixels; full-population first and
second moments and threshold counts are retained in the preflight artifact.

Define `x = abs(E_t)/C`, with inherited `C=4.960109710693359`. The source
distribution is finite, has exact-zero fraction `0.208134238141`, and has:

| source `x` condition | fraction |
|---|---:|
| `x < 0.1` | 0.398770136 |
| `x < 0.25` | 0.451318211 |
| `x < 0.5` | 0.499645612 |
| `x < 1` | 0.561283008 |
| `x >= 1` | 0.438716992 |
| `x >= 2` | 0 |
| `x >= 5` | 0 |

The systematic `x` quantiles are q10 `0`, q25 `0.0001701549`, q50
`0.5022023`, q75 `1.0000003`, q90 `1.0000004`, q95 `1.0000004`, and q99
`1.0000005`. The clamp therefore has a real source plateau; it is not merely
an abstract asymptote.

### Candidate maps

All candidates preserve the full signed teacher target `E_t`; only the
detached source-example loss importance changes.

| candidate | exact map | target | source full mean | source exact-one | effective fraction | Gini | initial-gradient mass proxy |
|---|---|---|---:|---:|---:|---:|---:|
| A: P33 clamp | `clip(x,0,1)` | `E_t` | 0.515119 | 0.438717 | 0.549288 | 0.478172 | 0.510859 |
| B: soft tanh | `tanh(x)` | `E_t` | 0.402036 | 0 | 0.563568 | 0.465266 | 0.397787 |
| C: rational | `x/(1+x)` | `E_t` | 0.269451 | 0 | 0.577860 | 0.453398 | 0.265495 |

The mass proxy is `w * min(abs(E_t), 1)` at a zero student effect, i.e. the
absolute SmoothL1 slope before adapter Jacobians. Relative to A, B retains
`77.8664%` of this aggregate source mass and C retains `51.9704%`. B is the
smallest non-null change that removes the hard source plateau while retaining
more high-actionability pressure than C. These are source diagnostics, not
held performance estimates.

At fixed `x`, the maps behave as follows:

| `x` | A clamp | B tanh | C rational |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 0.01 | 0.01 | 0.0099997 | 0.0099010 |
| 0.1 | 0.1 | 0.0996680 | 0.0909091 |
| 0.25 | 0.25 | 0.2449187 | 0.2000000 |
| 0.5 | 0.5 | 0.4621172 | 0.3333333 |
| 1 | 1 | 0.7615942 | 0.5 |
| 2 | 1 | 0.9640276 | 0.6666667 |
| 5 | 1 | 0.9999092 | 0.8333333 |
| 10 | 1 | 1.0000000* | 0.9090909 |

`*` is the finite-precision display of the mathematical tanh asymptote;
the locked source has no `x>=2` mass. Derivatives are: clamp `1` below one
and `0` above one; tanh `sech²(x)` (at `x=1`, `0.4199743`); rational
`(1+x)^-2` (at `x=1`, `0.25`).

Category means vary because the inherited effect proxy varies by category,
but no candidate introduces a category-specific rule. Across the eleven
source categories, mean `x` ranges `0.399029–0.602389`, mean tanh weight
`0.313833–0.472495`, and fraction `x>=1` ranges `0.333333–0.500411`.

## 4. Optimization-mass and identifiability result

The approximate source gradient mass used `g=w*abs(rho'(E_s-E_t))` with
`E_s=0` and inherited SmoothL1 beta 1. It is deliberately labeled a
zero-student source proxy, not a trained-gradient measurement:

| map | mean `g` | q99 `g` | effective fraction | Gini | top-10% mass | max |
|---|---:|---:|---:|---:|---:|---:|
| clamp | 0.510859 | 1.000000 | 0.540800 | 0.484191 | 0.195753 | 1.000000 |
| tanh | 0.397787 | 0.761594 | 0.552683 | 0.472837 | 0.191463 | 0.761594 |
| rational | 0.265495 | 0.500000 | 0.562864 | 0.463821 | 0.188332 | 0.500000 |

All maps are monotonic, finite, bounded, and preserve student radial
identifiability because the student effect is not normalized and the target
remains `E_t`. The selected tanh map changes importance only. At `w=0`, P35
intentionally supplies zero direct loss gradient, just as P33 does; this is
the stated hypothesis and is not the P34 zero-target semantics.

## 5. Narrow prior-art findings

The targeted review stopped after the candidate ranking stopped changing. The
records below use `problem → mechanism → equation → relevance → limitation`.

1. **Confidence Conditioned Knowledge Distillation** (Mishra & Sundaram,
   2021): heterogeneous teacher reliability → sample-specific loss and target
   conditioning → confidence enters CCKD-L/CCKD-T → direct precedent for
   source-dependent distillation importance and target variants → classification
   confidence and labels, not native-relative anomaly effects. [Primary paper](https://arxiv.org/abs/2107.06993)
2. **Not All Knowledge Is Created Equal: Mutual Distillation of Confident
   Knowledge** (Li et al., 2021): unreliable teacher knowledge → static or
   progressive confidence selection → entropy threshold selects knowledge
   before distillation → supports asking whether all teacher evidence should
   have equal influence → hard/progressive thresholds, mutual training, and
   noisy-label classification. [Primary paper](https://arxiv.org/abs/2106.01489)
3. **Distilling Knowledge From a Deep Pose Regressor Network** (Saputra et
   al., ICCV 2019): variable regression-teacher reliability → teacher-loss
   confidence weights attentive imitation → schematically `L=w_t L_reg` →
   close continuous confidence-weighted regression precedent → pose-specific
   confidence and auxiliary attentive hints, not anomaly residual intervention.
   [Primary paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Saputra_Distilling_Knowledge_From_a_Deep_Pose_Regressor_Network_ICCV_2019_paper.html)
4. **Selective Knowledge Distillation for Neural Machine Translation** (Wang
   et al., ACL-IJCNLP 2021): some training examples' knowledge harms the
   student → batch/global sample selection → select subsets for KD → direct
   precedent for optimization allocation based on sample usefulness → NMT,
   selection rather than this bounded pixelwise continuous map. [Primary paper](https://aclanthology.org/2021.acl-long.504/)
5. **Focal Loss for Dense Object Detection** (Lin et al., ICCV 2017): dense
   easy examples overwhelm optimization → dynamically down-weight easy
   examples → `FL(p_t)=-(1-p_t)^gamma log(p_t)` → foundational precedent that
   gradient allocation can matter independently of output-target geometry →
   supervised classification and a tuned gamma, not KD or anomaly residuals.
   [Primary paper](https://openaccess.thecvf.com/content_ICCV_2017/html/Lin_Focal_Loss_for_Dense_Object_Detection_ICCV_2017_paper.html)
6. **Uninformed Students** (Bergmann et al., CVPR 2020): one-class anomaly
   detection → teacher-student feature discrepancy → `A(x)=d(f_T(x),f_S(x))`
   → direct teacher-student anomaly precedent → discrepancy scoring rather
   than native-relative functional-effect transfer. [Primary paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Bergmann_Uninformed_Students_Student-Teacher_Anomaly_Detection_With_Discriminative_Latent_Embeddings_CVPR_2020_paper.html)
7. **Deep Anomaly Detection by Residual Adaptation** (Deecke et al., 2020):
   adapt a pretrained representation → residual correction `f=f_0+r_theta`
   → direct adapter context → no source-only confidence weighting around a
   strong native detector. [Primary paper](https://arxiv.org/abs/2010.02310)

### Prior-art overlap position

Importance weighting, confidence-conditioned distillation, sample selection,
residual adaptation, and teacher-student anomaly detection all have prior art.
The review found no basis for a novelty claim for `tanh`, confidence weighting,
or their components. It did not verify the exact scoped combination of cached
native-relative `9x9 -> deployed functional effect` transfer, a
source-only absolute-effect weight, and zero inference overhead. That is an
overlap boundary only, not a novelty assertion.

## 6. Candidate hypotheses

### Candidate A — P33 hard-clamp baseline

`w=clip(abs(E_t)/C,0,1)` and
`L=mean(w*SmoothL1(E_s,stop_gradient(E_t),beta=1))`.

This is the null/simple baseline. It directly tests sample importance and
retains the full signed target. Its limitation is the observed 43.87% source
hard plateau. Cheapest falsification: source-only saturation audit plus a
future single locked comparison against the native endpoints.

### Candidate B — selected soft actionability weighting

`w=stop_gradient(tanh(abs(E_t)/C))` and
`L=mean(w*SmoothL1(E_s,stop_gradient(E_t),beta=1))`.

This tests whether preserving the full functional target while removing hard
weight saturation improves gradient allocation. Expected detection effect is
better preservation of P33's pAP gain and a possible pAUROC recovery; expected
residual behavior is not prescribed to be sparse. Teacher direction matters
only insofar as it remains the full target, not as a separate fidelity gate.
It has one objective, zero new tuned parameters, no category-specific state,
training-only O(N) scalar work, and zero inference overhead. Main failure mode:
the absolute deployed effect is not a useful importance signal, or softening
reduces useful high-effect gradient mass. Cheapest falsification: fixed-point,
source-mass, radial, heavy-tail, and mixed-batch preflight, followed by the
single future Stage 2 gate.

### Candidate C — rational soft actionability weighting

`w=abs(E_t)/C /(1+abs(E_t)/C)` and the same full-target objective.

It is bounded and non-saturating, but its source weight at `x=1` is only `0.5`
and it retains about half of the P33 initial-gradient mass proxy. That is a
larger unsupported attenuation of actionable examples than B. It is retained
as the maximum comparison alternative, not selected.

| Candidate | Mechanism match | Full target | Objectives | New params | Train overhead | Inference overhead | Overconstraint risk | Falsifiability |
|---|---|---:|---:|---:|---:|---:|---|---|
| A clamp/P33 | high; null baseline | yes | 1 | 0 | baseline | 0% | low | high |
| B tanh | highest for saturation-is-limiting hypothesis | yes | 1 | 0 | O(N), negligible | 0% | low | high |
| C rational | medium; stronger attenuation | yes | 1 | 0 | O(N), negligible | 0% | low | high |

## 7. Selected next hypothesis

`SELECTED_P35_HYPOTHESIS = SOFT_ACTIONABILITY_WEIGHTED_FUNCTIONAL_TRANSFER`

### Causal claim

P33's improvement is more plausibly attributable to reallocating optimization
importance toward stronger source evidence than to producing sparse outputs.
The inherited clamp, however, assigns identical maximum importance to 43.87%
of source pixels. Replacing only that hard clamp with the parameter-free,
bounded, monotonic `tanh(x)` map should preserve the full signed functional
teacher target while providing continuous high-effect ordering and avoiding
the larger target/gradient attenuation of P34 and Candidate C.

### Why this does not repeat P30/P32/P33/P34

- P30's radial freedom is avoided: student effect magnitude is not normalized.
- P32's global unweighted functional transfer is not repeated.
- P33's useful full target is retained, while only the saturated importance map changes.
- P34's target shaping `wE_t` is explicitly excluded; P35 does not shrink the answer.
- No additional loss, gate, threshold, category parameter, inference module, or teacher-at-inference path is introduced.

## 8. Cheapest falsification sequence

1. Re-run the deterministic source-only script and verify the frozen source
   manifest, no held reads, and the candidate statistics.
2. Verify analytically and synthetically that all three maps are finite,
   monotonic, bounded, target-preserving, and radially identifiable; reject if
   the selected map needs a new constant or a target change.
3. Run import/objective/reference parity, the P34 reporting regression, a
   cached one-step smoke, checkpoint reload, and short profiles.
4. Only a separately authorized, one-attempt P35 Stage 2 could test the
   locked pAP/pAUROC endpoints; no such attempt is authorized here.

## 9. Overconstraint, cost, and authorization

The selected design is one mechanism, one SmoothL1 objective, zero new tuned
scalars, zero learned parameters, zero category-specific parameters, zero
inference overhead, and one vectorized `tanh` per deployed teacher-effect
element. It is intentionally a change in optimization importance only.

The future P35 Stage 2 would inherit the candle LOCO 20-epoch, batch-1,
39,240-step FP32 AdamW schedule and compare against frozen P31/native,
P30R1, P32, and P33. Detection and tail-safety endpoints are gates; residual
support is descriptive, not a required P30R1-like sparsity target.

This decision authorizes preparation of the P35 preflight, preregistration,
minimal implementation, and engineering qualification only. It authorizes
no P35 scientific UUID, held prediction, Stage 2, Stage 3, subset, or full run.

P35 scientific Stage 2 attempts = `0`

P35 Stage 3 attempts = `0`

P35 full runs = `0`

P35 held tuning iterations = `0`

`P35_RESEARCH_DECISION_COMPLETE`
