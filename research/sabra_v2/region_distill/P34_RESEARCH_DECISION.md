# P34 Research Decision — Explicit Actionability Target

Status: `P34_RESEARCH_DECISION_COMPLETE`

Protocol identifier: `P34`. This is an offline decision artifact. It does not
create a scientific execution UUID and does not authorize a P34 Stage 2 run.

## 1. Entry and authoritative P33 result

P33 terminated at `P33_STAGE2_SCIENTIFIC_STOP`. Its final evidence commit was
`f83e8d7053b2abe78cf92b4aadb4043ad673c6a9`. P33 used one objective,
source-only actionability weighting, and no inference-time module.

| method | pAP | pAUROC |
|---|---:|---:|
| P31/native | 0.514140304931 | 0.980667143514 |
| P30R1 | 0.511513734224 | 0.980534708954 |
| P32 | 0.510351502947 | 0.971460700418 |
| P33 | 0.519395095936 | 0.978184288830 |

P33 improved pAP over P32 and native, but did not recover native pAUROC.
Its residual q99 and normal-score shift remained safe. The relevant frozen
selectivity observations were:

- P33 effective residual support was `0.962760`, Gini `0.069176`, and
  inherited-threshold support was `0.999074`.
- P30R1 historical support was `0.111358`, with effective support `0.056409`
  in the comparable forensic definition.
- P33 source actionability weights had mean `0.515119`, median `0.501540`,
  q90/q95/q99 all `1.0`, and exact-zero fraction `0.208134`.
- P33 retained the useful P30R1 locations descriptively, but also added broad
  intervention. This is a selectivity failure, not a radial explosion.

The P33 held observations are used here only as frozen post-run evidence for
forensic motivation. No P34 held result, label, mask, or metric was accessed.

## 2. Forensic question and gradient proof

The causal question is whether P33 failed because its actionability signal
controlled loss importance but did not specify the output on abstaining
locations.

Let `N` be the number of effect pixels and define the SmoothL1 derivative:

```text
psi_beta(u) = sign(u) * min(abs(u) / beta, 1)
```

P33 is:

```text
L_P33 = mean(w * SmoothL1(E_s - E_t; beta=1))
dL_P33/dE_s = w * psi_1(E_s - E_t) / N
```

P34 is:

```text
T = stop_gradient(w * E_t)
L_P34 = mean(SmoothL1(E_s - T; beta=1))
dL_P34/dE_s = psi_1(E_s - w*E_t) / N
```

Therefore:

| condition | P33 loss-weighting | P34 target shaping |
|---|---|---|
| `w=0`, `E_s!=0` | exact zero gradient; no direct correction | nonzero gradient toward target zero |
| `w=0`, `E_s=0` | zero gradient | stable zero optimum |
| `w=0.1` | 0.1-scaled gradient toward `E_t` | gradient toward the attenuated target `0.1 E_t` |
| `w=0.5` | 0.5-scaled gradient toward `E_t` | target is `0.5 E_t` with ordinary loss strength |
| `w=1` | ordinary functional transfer | identical functional target |

Under the actual source rule, `w=0` normally accompanies `E_t=0`. The
decoupled synthetic case `w=0, E_t!=0` isolates the operator algebra. The
important actual case is still `E_t=0, E_s!=0`: P33 supplies no restoring
gradient, while P34 supplies the gradient of a zero target. Because adapter
parameters are shared, active examples can move nominal examples away from
zero unless nominal examples explicitly train toward zero.

The proof is implemented in
[`P34_PREFLIGHT_FALSIFICATION.json`](P34_PREFLIGHT_FALSIFICATION.json).

## 3. Source-only target and saturation audit

The audit used only the locked Tier-B candle source tensor
`/workspace/p27r1_cache_v1/tier_b/candle/teacher_region.npy`; it did not open
held data or run a neural model. The exact source population contains
`526451688` deployed effect pixels.

Using the inherited source-derived threshold `C/100`, where
`C=4.960109710693359`:

| quantity | raw `E_t` | shaped `w E_t` |
|---|---:|---:|
| exact-zero fraction | 0.208134 | 0.208134 |
| near-zero `<=1e-6` | 0.211298 | 0.260840 |
| meaningful support `>C/100` | 0.686707 | 0.601230 |
| effective-support fraction (systematic sample) | 0.549465 | 0.502971 |
| Gini (systematic sample) | 0.477993 | 0.514036 |
| q50 absolute magnitude (sample) | 2.494714 | 1.254730 |
| q90/q95/q99 absolute magnitude (sample) | 4.960112 / 4.960112 / 4.960112 | 4.960112 / 4.960112 / 4.960112 |

The shaped target does not create more exact zeros under this rule; it creates
the same zero locations and gives every such location an explicit zero
target. It also attenuates low-actionability nonzero targets. It is not an
objective to maximize sparsity: the source target retains nontrivial signal,
with target-to-raw absolute mass ratio `0.937819` and RMS ratio `0.980240`.

The weight saturation audit found:

- exact `w=1`: `0.438717` of source pixels;
- `w>0.75`: `0.468222`;
- `w>0.9`: `0.452384`;
- pre-clamp `abs(E_t)/C >= 1`: `0.438717`.

Saturation is a limitation of the inherited actionability proxy, but changing
the transform now would simultaneously change actionability and target
semantics. That would not cleanly test the P33 failure mechanism.

## 4. Preflight falsification

The deterministic suite covered exact zero, near zero, `0.01x`, `0.1x`, `1x`,
`10x`, `100x`, sign reversal, sparse actionable support, heavy tails,
mixed-scale batches, one extreme outlier, all-abstain, all-active, and a
teacher-zero case. It checked loss, gradient norms, maximum gradient,
finite-ness, target magnitude, and batch dominance.

All gates passed:

- zero-actionability restoring gradient: pass;
- zero optimum: pass;
- full actionability remains ordinary functional transfer: pass;
- intermediate target remains continuous and identifiable: pass;
- heavy-tail finite behavior: pass;
- mixed-batch gradient dominance: pass;
- meaningful source support decreases without near-total collapse: pass;
- radial identifiability remains because the student is not normalized: pass;
- no new tuned scalar is required: pass.

No all-zero collapse claim is being made from synthetic data. The preflight
shows that active targets remain substantial; whether shared adapter learning
preserves useful action is the one future scientific question.

## 5. Narrow prior-art findings

The targeted search stopped when the candidate ranking stopped changing. Each
entry is `problem → mechanism → equation → overlap → limitation`.

1. **Confidence Conditioned Knowledge Distillation (Mishra & Sundaram,
   2021):** unreliable teacher → confidence-dependent loss or target →
   `L=λL_KD+(1−λ)L_CE`, and a confidence-conditioned target formed by mixing
   teacher and label distributions → direct precedent that confidence can
   shape the transferred target, not only its loss weight → uses labels and a
   classification probability simplex; it is not a native-relative anomaly
   residual target. [Primary paper](https://arxiv.org/abs/2107.06993)

2. **Not All Knowledge Is Created Equal / Confidence-Driven Masking (Li et
   al., 2021):** unreliable knowledge → retain or mask teacher information by
   confidence, with static/progressive thresholds → selective transfer rather
   than uniform imitation → relevant to actionability selection → hard
   thresholds, extra schedule choices, and a noisy-label classification
   setting. [Primary paper](https://arxiv.org/abs/2106.01489)

3. **Distilling Knowledge From a Deep Pose Regressor Network (Saputra et al.,
   2019):** regression teacher reliability varies → teacher-confidence-weighted
   attentive regression transfer, schematically `L=w_t L_reg` → precedent for
   continuous confidence weighting in regression → confidence is defined for
   pose regression and does not create a native/no-op target. [Primary
   paper](https://www.qmac.ox.ac.uk/files/11078/ICCV19_Distilling_Knowledge_From_a_Deep_Pose_Regressor_Network.pdf)

4. **Localization Distillation for Dense Object Detection (Zheng et al.,
   2022):** ordinary feature imitation misses localization knowledge → select
   valuable spatial regions and distill detector outputs, expressible as a
   region-weighted output loss → supports downstream-aware spatial transfer →
   detector-specific region selection and no anomaly residual abstention
   target. [Primary paper](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Localization_Distillation_for_Dense_Object_Detection_CVPR_2022_paper.html)

5. **SelectiveNet (Geifman & El-Yaniv, 2019):** prediction with a reject
   option → learned rejector with coverage-constrained selective risk,
   `coverage=E[g(x)]` and risk weighted by `g` → formalizes abstention as an
   explicit action decision → requires a learned gate and labeled coverage
   risk, so it is too elaborate for this one-objective frozen adapter.
   [Primary paper](https://proceedings.mlr.press/v97/geifman19a.html)

6. **Uninformed Students (Bergmann et al., 2020):** one-class anomaly
   detection → student–teacher discrepancy, `A(x)=d(f_T(x),f_S(x))` → direct
   teacher–student anomaly precedent → the discrepancy is the detection
   signal, not a selective native-relative correction target. [Primary
   paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Bergmann_Uninformed_Students_Student-Teacher_Anomaly_Detection_With_Discriminative_Latent_Embeddings_CVPR_2020_paper.html)

7. **Deep Anomaly Detection by Residual Adaptation (Deecke et al., 2020):**
   adapt a pretrained detector representation → residual adapter
   `f=f_0+r_theta` → direct residual-adaptation precedent → no explicit
   abstention or zero-target semantics around a strong frozen detector.
   [Primary paper](https://arxiv.org/abs/2010.02310)

### Prior-art overlap position

Confidence weighting, target conditioning, selective transfer, output
distillation, abstention, teacher–student anomaly detection, and residual
adaptation all have prior art. Target-shaped confidence transfer already
exists in classification and regression forms. The targeted search did not
verify the exact cached native-relative `9x9 -> deployed functional effect`
target `stop_gradient(clamp(abs(E_t)/C,0,1)*E_t)` in anomaly residual
adaptation. This is an overlap boundary, not a novelty claim.

## 6. Candidate mechanisms

### Candidate A — `EXPLICIT_ACTIONABILITY_TARGET_FUNCTIONAL_TRANSFER`

```text
w = stop_gradient(clamp(abs(E_t)/C, 0, 1))
T = stop_gradient(w * E_t)
L = mean(SmoothL1(E_s, T; beta=1))
```

This directly tests whether low actionability must be an explicit no-op
target. It has one objective, zero new tuned parameters, preserves student
scale, and has zero inference overhead. It passed all preflight gates.

### Candidate B — `RESIDUAL_TARGET_INTERPOLATION`

An interpolation between no correction and the teacher functional correction
is exactly `T=(1-w)*0+w*E_t=wE_t` in the deployed effect space. It is therefore
not a distinct candidate and is merged with Candidate A. Calling it a second
method would only rename the same equation.

### Candidate C — `UNSATURATED_BOUNDED_ACTIONABILITY_TARGET`

An analytic alternative is `w'=q/(1+q)`, `q=abs(E_t)/C`, with `T'=w'E_t`.
It is bounded, monotone, zero-preserving, and introduces no scalar. In the
source sample it retains only about `0.493` of raw target RMS and `0.477` of
raw absolute mass. It changes the P33 actionability transform and the target
at once, so it cannot isolate whether P33 failed because of weighting versus
target semantics. It is rejected for this protocol; the saturation result is
retained as a limitation and possible later question.

| Candidate | Explicit zero gradient | Action preserved | One objective | New tuned params | Radial identifiable | Source-only | Inference overhead | Main risk |
|---|---|---|---:|---:|---|---|---:|---|
| A explicit shaped target | yes | yes at `w=1`; attenuated continuously below | 1 | 0 | yes | yes | 0% | proxy may still mark too many locations actionable |
| B interpolation | yes | same as A | 1 | 0 | yes | yes | 0% | duplicate equation, no independent test |
| C unsaturated transform | yes | weaker for large effects | 1 | 0 | yes | yes | 0% | confounds transform saturation with target semantics |

Candidate A ranks first by causal fit, simplicity, source-only identifiability,
falsifiability, runtime, and publishability. Candidate B is not separately
implemented. Candidate C is not the clean next experiment.

## 7. Selected next hypothesis

`SELECTED_NEXT_HYPOTHESIS = EXPLICIT_ACTIONABILITY_TARGET_FUNCTIONAL_TRANSFER`

P33 failed to restore selectivity because loss weighting suppresses gradients
on low-actionability samples without explicitly restoring their residual to
zero. Shaping the functional correction target by the same bounded,
source-only actionability signal should provide an explicit abstention target
while preserving the full functional correction on high-actionability
samples.

This is a causal target-semantics test, not a metric repair, sparsity penalty,
teacher-direction recovery, hard gate, or added loss stack. P33 evidence
motivates it because P33 improved pAP but retained nearly all intervention
locations and its zero-weight fraction did not yield sparse final residuals.
The hypothesis would be falsified by a future locked result in which P34 does
not reduce dense intervention relative to P33, or reduces it only by
collapsing actionable target signal, or fails the native detection/safety
criteria. Those criteria are frozen in the preregistration before any P34
held result.

## 8. Future falsification gates and authorization boundary

The future protocol must test both action preservation and abstention. Before
any scientific marker, the exact target, gradient, source distribution,
reference parity, and engineering path must pass. The future held gate will
compare pAP/pAUROC with native and use the inherited `C/100` diagnostic only
as a source-derived relative mechanism measure; it will not require the
historical P30R1 support percentage.

P34 implementation and engineering qualification are authorized after this
decision and before a future scientific attempt. The present decision itself
performed no optimizer step and created no scientific run.

```text
P34 scientific Stage 2 attempts = 0
Stage 3 attempts               = 0
full runs                      = 0
held tuning iterations         = 0
new CLIP forwards              = 0
new Phase2B forwards           = 0
teacher forwards               = 0
cache rebuilds                 = 0
```

`P34_RESEARCH_DECISION_COMPLETE`
