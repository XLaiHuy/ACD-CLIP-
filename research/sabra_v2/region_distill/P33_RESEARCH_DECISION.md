# P33 Research Decision — Selective Actionability After P32 Failure

Status: `P33_RESEARCH_DECISION_COMPLETE`

Protocol identifier: `P33`. This is a frozen/offline decision artifact. It is
not a scientific execution marker and does not authorize a P33 Stage 2 run.

## 1. Entry and authoritative result

P32 terminated at `P32_STAGE2_SCIENTIFIC_STOP` with attempt
`1818b1ef-9afc-41ad-972c-ee5288ee1286`; its final synchronized evidence is
commit `74d88aeb2c72e5a675984c802fb36acbbbe8dda1`. The paired frozen payloads
have identical image order and identical native maps. No neural model was
run in this phase.

| method | pAP | pAUROC |
|---|---:|---:|
| P31/native | 0.514140304931 | 0.980667143514 |
| P30R1 | 0.511513734224 | 0.980534708954 |
| P32 functional margin effect | 0.510351502947 | 0.971460700418 |

P32 was safe radially: residual q99 was `4.963814287186`, normal-score q99
shift was `0.000005382090`, there was no scale explosion or normal-score
saturation, and the post-run audit passed. The failure is therefore a
downstream/ranking failure, not an obvious numerical-safety failure.

## 2. Why P32 failed

P32 matched the deployed functional margin effect everywhere the objective
could see it. That removed raw-vector direction as a target, but it did not
declare where intervention should exist. At the inherited P30R1 coordinate
threshold `0.0496010971069336`, P32 had `87.1481%` active residual
coordinates versus `11.1358%` for P30R1. Its residual effective support was
`25536.73` (`52.5447%`) versus `2741.49` (`5.6409%`), with Gini `0.5051`
versus `0.9251`; its top 10% carried only `21.45%` of residual mass versus
`96.11%` for P30R1.

The deployed score effect was not large everywhere: more than `99.3%` of
both methods' pixels were below `1e-4`, and both effects placed about
`99.996%` of absolute mass in their top 1%. But P32's micro-effect floor was
much denser: only `3.32%` of its score effects were at most `1e-10`, versus
`77.53%` for P30R1; the P32 median absolute effect was `3.26e-9` versus
`3.12e-11`. Thus “tiny” does not mean “absent”: the low-level effect tail
can perturb a detector whose native score gaps are extremely small.

P32 had lower native-vs-candidate rank agreement (fixed-stride pooled
Spearman `0.9324` versus P30R1 `0.9749`; mean per-image Spearman `0.7355`
versus `0.9072`) and greater rank displacement (median `0.1163` versus
`0.01635` of image pixels). However, adjacent native-order flips were about
`50%` for both methods, and predefined native-gap bins did not show a
P32-specific excess in the smallest gaps. Near-tie sensitivity is therefore
a consequence/secondary amplifier, not the isolated primary mechanism.

The residual support comparison is decisive about the shape of the failure:
P30R1 active support was `97.01%` contained in P32 support, while P32 active
support was only `12.40%` contained in P30R1 support (Jaccard `0.1235`). P32
mostly retained the sparse support and added broad extra support. At a
deployed effect threshold of `1e-4`, support Jaccard was `0.8780`, so the
problem is best described as dense residual/micro-effect expansion rather
than wholesale relocation of the useful tail.

The descriptive held-mask check does not rescue a global transfer rule:
P30R1 absolute effect mass was `291.875x` anomaly-area enriched, but P32 was
also enriched at `315.155x`. P32's anomaly enrichment is not weak; enrichment
alone is not sufficient to preserve the native ranking. Native remains the
strongest locked control.

## 3. Required hypothesis ranking

1. `H1_LOST_SELECTIVITY_DENSE_MICRO_CORRECTION` — directly supported by
   meaningful residual support expansion and a denser micro-effect floor.
2. `H4_SUPPORT_MISMATCH` — supported as support expansion; P32 retained most
   P30R1 support but added a large extra set.
3. `H5_ACTIONABILITY_NOT_MAGNITUDE` — motivated by P30R1's sparse intervention
   and P32's unconditioned effect matching, but not yet causally isolated.
4. `H2_WRONG_FUNCTIONAL_TARGET` — possible, but P32 did not separate an
   invalid target from an unselective application of the target.
5. `H3_NEAR_TIE_RANKING_SENSITIVITY` — rank displacement increased, but the
   conditional flip analysis does not isolate a P32-specific near-tie effect.

The machine-readable values and inventory are in
[`P33_FORENSIC_ANALYSIS.json`](P33_FORENSIC_ANALYSIS.json). The analysis
implementation is
[`p33_selective_actionability.py`](../../../tools/sabra_v2/forensics/p33_selective_actionability.py).

## 4. Source-only actionability audit

The immutable Tier-B source union contains 2,162 unique samples across
23,782 exposures. Five initial descriptors were examined without held
outcomes and without a learned classifier:

| descriptor | frozen observation | decision |
|---|---|---|
| pixel `abs(deployed teacher effect)` | 21.38% exact zero in a fixed spatial sample; q25 `0.000610`, q50 `2.62149`, q75 `4.96011` | retain as the only spatially useful operational proxy |
| sample deployed-effect RMS | q01 `0.16094`, q50 `1.98916`, q99 `4.96011`; category median ratio `4.48` | usable, but category dispersion is a limitation |
| sample teacher-region RMS | q50 `2.05686`; category median ratio `4.34` | redundant with effect magnitude |
| raw-region support above inherited epsilon | mean `0.57745`; category median ratio `5.79` | reject as category-sensitive hard support |
| deployed-effect top-10% mass | median `0.31118`; category median ratio `6.56` | concentration descriptor, not a stable actionability rule |

Cross-stage teacher consistency, multi-view stability, and an independent
calibrated teacher confidence are unavailable in the frozen source cache.
The absolute deployed teacher effect identifies an operational “the teacher
asks for an effect here” quantity. It does not establish that the effect is
useful on held anomalies; that is precisely what the next clean experiment
would test.

## 5. Narrow prior-art findings

The search was stopped after the ranking stopped changing. Each entry is
`problem → mechanism → equation → relevance → limitation`.

1. **Confidence Conditioned Knowledge Distillation (Mishra & Sundaram,
   2021):** unreliable teacher predictions → sample-specific confidence
   weighting or targets → `L=λL_KD+(1−λ)L_CE`, with `λ` derived from teacher
   confidence → direct precedent for weighting transfer by teacher evidence
   → uses the correct label to define confidence and is classification/
   regression-specific. [Primary paper](https://arxiv.org/abs/2107.06993)

2. **Not All Knowledge Is Created Equal / CMD (Li et al., 2021):** noisy or
   unreliable knowledge → select confident knowledge with static/progressive
   thresholds → transfer is gated when confidence exceeds a threshold →
   direct precedent for asking which teacher knowledge should be transferred
   → hard thresholds, mutual models, label-noise setting, and extra schedule
   choices. [Primary paper](https://arxiv.org/abs/2106.01489)

3. **Distilling Knowledge From a Deep Pose Regressor Network (Saputra et
   al., 2019):** regression teacher is not uniformly reliable → teacher-loss
   confidence weights an attentive imitation loss, schematically
   `L_AIL=w_t L_reg` → precedent for continuous confidence-weighted regression
   transfer → pose-regression confidence and auxiliary attentive-hint setting,
   not native-relative anomaly intervention. [Primary paper](https://www.qmac.ox.ac.uk/files/11078/ICCV19_Distilling_Knowledge_From_a_Deep_Pose_Regressor_Network.pdf)

4. **Localization Distillation for Dense Object Detection (Zheng et al.,
   2022):** feature imitation misses localization knowledge → output/logit
   distillation on valuable regions, expressible as a region-weighted output
   loss `Σ_{p∈R}w_p L_out(p)` → precedent for downstream-aware spatial
   selection → detection boxes/regions and task-specific selection; no
   anomaly residual study. [Primary paper](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Localization_Distillation_for_Dense_Object_Detection_CVPR_2022_paper.html)

5. **SelectiveNet (Geifman & El-Yaniv, 2019):** prediction with abstention →
   learned reject gate with coverage-constrained risk →
   `coverage=E[g(x)]`, `risk=E[g(x)ℓ]/E[g(x)]` → formalizes why abstention
   must have an explicit coverage/utility semantics → adds a learned gate,
   coverage machinery, and held-label risk; rejected for the minimal P33
   scope. [Primary paper](https://proceedings.mlr.press/v97/geifman19a.html)

6. **Uninformed Students (Bergmann et al., 2020):** one-class pixel anomaly
   detection → student regresses a pretrained teacher and discrepancy scores
   anomalies → `A(x)=d(f_T(x),f_S(x))` → direct teacher–student anomaly
   precedent → discrepancy is the inference signal, not a native-relative
   selective correction. [Primary paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Bergmann_Uninformed_Students_Student-Teacher_Anomaly_Detection_With_Discriminative_Latent_Embeddings_CVPR_2020_paper.html)

7. **Student-Teacher Feature Pyramid Matching (Wang et al., 2021):**
   multi-scale anomaly detection → feature-pyramid matching and discrepancy
   scoring → multi-level `d(f_T,f_S)` anomaly maps → confirms anomaly
   teacher–student practice → no native-relative residual actionability or
   selective transfer. [Primary paper](https://arxiv.org/abs/2103.04257)

8. **Deep Anomaly Detection by Residual Adaptation (Deecke et al., 2020):**
   adapt pretrained representations to anomaly detection → residual
   corrections `f=f_0+r_θ` → direct residual-adaptation precedent → relevant
   to the adapter form → does not define abstention/actionability weighting
   around a strong native detector. [Primary paper](https://arxiv.org/abs/2010.02310)

### Prior-art overlap audit

The components—confidence weighting, hard selection, spatial output
distillation, abstention, teacher–student anomaly scoring, and residual
adaptation—already exist separately. The targeted set did not reveal this
exact combination: a cached native-relative 9×9 region correction, mapped to
the deployed margin effect, with a source-only continuous absolute-effect
weight and no inference-time gate. This is a scoped overlap statement, not a
novelty claim.

| question | finding |
|---|---|
| exact mechanism already exists? | components exist separately; exact P33 combination not found in this targeted set |
| anomaly detection? | yes for teacher–student discrepancy and residual adaptation; not this intervention |
| teacher–student anomaly detection? | yes, but reviewed methods score feature discrepancy rather than selective native-relative correction |
| residual/logit intervention? | output/logit distillation and residual adaptation exist; exact cached native-relative effect weighting remains unverified |
| extra inference cost? | P33 adds none; weight is training-only |
| multiple weighted losses? | several precedents use gates, coverage, pairwise, or auxiliary terms; P33 uses one objective |
| category-specific parameters? | not required and forbidden for P33 |

## 6. Candidates and comparison

Let `D` be the frozen 9×9→518×518 deployment transform,
`E_s=D(mean_stage(r_s))`, `E_t=D(r_t)`, and
`C=4.960109710693359`, the inherited correction scale. All teacher-derived
quantities are detached.

### Candidate A — continuous actionability weighting

`w_A=clip(abs(E_t)/C,0,1)` and

```text
L_A = mean( w_A * SmoothL1(E_s, stop_gradient(E_t), beta=1.0) )
```

This retains the signed teacher effect as the target, but attenuates learning
where the teacher effect is near zero. It has one objective, no new learned
module, no new tuned scalar, and no inference-time cost.

### Candidate B — abstention-aware functional target

Use the same `w_A`, but target `w_A E_t`:

```text
L_B = mean SmoothL1(E_s, stop_gradient(w_A * E_t), beta=1.0)
```

This explicitly drives a null target toward zero, but also shrinks moderate
teacher effects. Synthetic all-abstain cases produce a restoring gradient on
arbitrary student output; this is safe numerically but repeats the radial
target-shrinkage risk that P30/P32 do not isolate.

### Candidate C — hard sparse support transfer

Use the inherited hard threshold on the teacher region, propagate its support
through `D`, and weight the same functional error:

```text
W_C = D(1[abs(r_t) > 0.0496010971069336])
L_C = mean( W_C * SmoothL1(E_s, stop_gradient(E_t), beta=1.0) )
```

It is stable and simple, but discontinuous, threshold-sensitive, and less
able to distinguish weak useful effects from numerical noise.

| Candidate | Selectivity causal fit | Source-only? | One objective? | Hyperparams | Radial identifiable? | Inference overhead | Main risk |
|---|---|---|---:|---:|---|---:|---|
| A continuous effect weighting | high | yes | 1 | 0 new | yes; target retained | 0% | effect magnitude is only an operational, not validated, utility proxy |
| B abstention-aware target | medium-high | yes | 1 | 0 new | weakened by target shrinkage | 0% | reintroduces magnitude attenuation and null-target pressure |
| C hard support transfer | medium | yes | 1 | 0 new | yes on selected support | 0% | discontinuity, threshold sensitivity, category/support mismatch |

## 7. Selected hypothesis

`SELECTED_P33_HYPOTHESIS = CONTINUOUS_ACTIONABILITY_WEIGHTED_FUNCTIONAL_TRANSFER`

### Causal claim

P32's functional margin target is not sufficient when applied globally. If the
teacher's absolute deployed margin effect is used as a bounded source-only
actionability weight, the student can learn the same signed downstream effect
while suppressing dense low-actionability micro-corrections. The hypothesis
tests *when/where to intervene*, not improved cosine, sign, Pearson,
Spearman, or a repaired internal metric.

### Why P29/P30/P30R1/P31/P32 motivate it

- P29 showed that multiple objectives can conflict.
- P30 showed that raw direction can coexist with disastrous downstream
  behavior and exposed radial risk.
- P30R1 recovered safe, near-native detection behavior with sparse,
  anomaly-enriched changes despite poor raw direction.
- P31/native remained slightly better than P30R1, making unnecessary
  intervention the central safety control.
- P32 removed raw direction and remained radially safe, but became meaningfully
  dense in residual space and produced broader rank displacement.

### Falsifier

A single future P33 candle test falsifies the hypothesis for this scope if the
actionability-weighted method fails the locked native non-inferiority endpoints
or health/safety gates, or if its frozen diagnostics show no reduction of
low-actionability support relative to P32. No threshold or coefficient may be
changed after seeing held results.

### Why A is simpler and safer

Candidate A is one bounded elementwise weight inside the existing one-objective
functional loss. It adds no gate, classifier, loss term, inference branch,
teacher forward, category parameter, or target normalization. It retains the
signed functional target wherever learning is active, so it does not repeat
P30's radial non-identifiability or Candidate B's systematic target shrinkage.

### New training required?

Yes, only for a future preregistered scientific comparison; no new scientific
training is performed in this phase. The engineering qualification, if run,
uses source cache only and is not a scientific attempt. A native/zero-adapter
control remains mandatory because native already explains the locked scope
nearly completely.

## 8. Cheapest falsification sequence

```text
symbolic bounded-gradient check
→ deterministic synthetic adversarial suite
→ source-cache weight/support and category-dispersion audit
→ production/reference parity
→ cached one-batch forward/backward/checkpoint smoke
→ only after a separately frozen preregistration: one clean P33 Stage 2 attempt
```

The synthetic suite includes exact zero, near zero, 0.01×/0.1×/1×/10×/100×
scales, sign reversal, 1% sparse support, heavy tails, mixed scales, one
extreme outlier, all abstain, and high-confidence intervention. Candidate A
was finite in all cases, with bounded SmoothL1 gradients and zero objective
gradient in the all-abstain case.

## 9. Overconstraint and novelty position

Overconstraint risk: `LOW` for Candidate A. The design has one mechanism, one
objective, zero new tuned hyperparameters, no category-specific parameter,
and zero inference overhead. The remaining risk is scientific—not
architectural: source effect magnitude may not be a reliable actionability
proxy. That risk is exposed by the native control and the one-attempt stop
rule.

Novelty position: no novelty claim. Prior art covers selective/confidence
weighted distillation, spatial output distillation, abstention, teacher–student
anomaly detection, and residual adaptation separately. P33 is a scoped causal
combination to test, not a claim that the ingredients are new.

## 10. Terminal counts for this decision phase

```text
new scientific Stage 2 attempts = 0
Stage 3 attempts                 = 0
full runs                        = 0
held tuning iterations          = 0
new CLIP forwards               = 0
new Phase2B forwards            = 0
new teacher forwards            = 0
cache rebuilds                  = 0
scientific UUIDs                = 0
```

`P33_RESEARCH_DECISION_COMPLETE`
