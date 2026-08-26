# P30R1 Targeted Research Report — Direction + Scale Stabilization

Status: **RESEARCH ONLY**. No P30R1/P31 implementation, training, scientific run, or execution marker was created by this research step.

Source baseline: branch `research/p29r1-fast-objective-forensic-v1`, P30 terminal commit `3cf91fb4325b9d3aea3b8a65d131cdea14d8ceba`.

## 1. P30 failure mechanism

### Verified local evidence

P30 Stage 2 correctly stopped on the preregistered candle gate. Relative to frozen P29:

- pixel AP: `0.144618064` vs `0.490503231`;
- pixel AUROC: `0.972904419` vs `0.970040297`;
- directional cosine: `0.736923574` vs `0.708549174`;
- sign agreement: `0.569567901` vs `0.565493827`;
- Pearson correction correlation: `0.378208786` vs `0.769957204`;
- Spearman correction agreement: `0.714233750` vs `0.717418156`;
- mean absolute residual: `1.542490326` vs `2.036682759`;
- residual q99 absolute: `25.929798947` vs `4.321676936`;
- normal score q99 shift: `0.998690784` vs `0.000001159`.

The mean residual did **not** globally inflate; instead the q99 tail exploded. This is important: P30 is not best described as a simple global-scale error. It produced a heavy-tailed / localized radial instability that was amplified by the deployment path.

### Objective-level localization

P30 flattens each staged residual to a 243-dimensional vector and normalizes teacher and student by their **own** per-sample RMS before computing the directional objective. For nonzero targets, if

`t = alpha * u`

and

`s = beta * v`,

then after self-normalization the loss primarily identifies `u` versus `v`; `beta` is largely free. Thus P30 can improve angular alignment while allowing student amplitude to collapse or explode.

A second blind spot is more specific: exact zero-teacher vectors are excluded from the directional average. Therefore a source sample for which the teacher says "no correction" need not directly push the student residual toward zero. That is especially dangerous for anomaly scoring because normal regions are precisely where uncontrolled positive correction is costly.

### Deployment amplification

The student residual is not merely an embedding used for cosine-distance anomaly scoring. It is upsampled and inserted through the unchanged **symmetric margin correction** into the frozen native two-class logits, then blurred/resized/averaged and softmaxed. Residual amplitude therefore has direct causal leverage over abnormal probability. A large positive tail can saturate normal-pixel anomaly probabilities even when the residual direction is reasonable.

This localizes the P30 failure to **B + C** in the research taxonomy:

- **B — residual/radial magnitude stability failed first**;
- **C — final anomaly-score calibration then failed as a downstream consequence**.

There is currently no local evidence that feature-level representation norm is the primary bottleneck.

### Mechanism conclusion

**P30 direction-only distillation was too scale-invariant for a deployment path whose semantics are scale-sensitive.**

The failure is stronger than "cosine is insufficient": P30 also removed the zero-target restoring force. The next method should retain the clean single-objective gradient path while making student radius identifiable and keeping zero-teacher samples active.

---

## 2. Targeted literature findings

The search was deliberately narrow. Five high-value prior-art lines were inspected rather than collecting a broad survey.

### 2.1 Wang et al. — Improving Knowledge Distillation via Regularizing Feature Norm and Direction (2023, arXiv:2305.17007)

Official repository: https://github.com/WangYZ1608/Knowledge-Distillation-via-ND

Mechanism: explicitly regularizes both **feature norm and direction** at the penultimate representation and is used as an add-on to multiple KD frameworks.

Relevance: directly establishes that "preserve norm + direction" is **not itself a novel contribution**. It also supports the general premise that direction-only information can be insufficient.

Limitation for SABRA: representation-level classification KD is not the same as a signed residual that is injected directly into anomaly logits. The method also does not isolate the P30 heavy-tail / normal-score saturation mechanism.

### 2.2 Deng & Li — Anomaly Detection via Reverse Distillation from One-Class Embedding (CVPR 2022)

Official repository: https://github.com/hq-deng/RD4AD

The official implementation trains using cosine similarity between encoder and decoder feature representations.

Relevance: strong negative-control prior art. It demonstrates that cosine/directional feature matching can work well for anomaly detection **when anomaly scoring itself is based on feature discrepancy**.

Why it does not resolve P30: SABRA deploys the learned residual as a signed logit-margin correction. Therefore radial scale has deployment semantics that RD4AD's cosine feature discrepancy does not share.

### 2.3 Wang et al. — Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection (STPM, 2021, arXiv:2103.04257)

Reference implementation inspected: https://github.com/hcw-00/STPM_anomaly_detection

Mechanism family: teacher-student feature matching across a pyramid; anomaly localization comes from teacher/student feature discrepancy.

Relevance: another anomaly-distillation precedent where representation discrepancy is the score signal rather than a directly injected correction amplitude.

Limitation for SABRA: does not answer how to preserve the physical scale of a signed residual used downstream as a logit shift.

### 2.4 Zhang et al. — DeSTSeg: Segmentation Guided Denoising Student-Teacher for Anomaly Detection (CVPR 2023)

Official repository: https://github.com/apple/ml-destseg

Mechanism: segmentation-guided denoising student-teacher architecture with additional components and training controls.

Relevance: confirms that richer feature/segmentation constraints are established anomaly-detection directions.

Limitation for SABRA: significantly heavier than the minimal P30R1 target and would weaken the clean causal question. Current local evidence does not justify escalating to feature-level or segmentation-level consistency.

### 2.5 Park et al. — Relational Knowledge Distillation (CVPR 2019, arXiv:1904.05068)

Official repository: https://github.com/lenscloth/RKD

Mechanism: transfers relational distance and angle information, with separate distance/angle ratios in the reference implementation.

Relevance: establishes prior art for combining geometric angular and radial/relational information.

Limitation for SABRA: it is not designed for anomaly residual calibration and requires multiple weighted relational terms; directly importing it would reintroduce extra interacting objectives and hyperparameters.

### Prior-art boundary

The inspected literature makes the following claims unsafe:

- "norm + direction distillation is novel" — **false / already established**;
- "angle + distance geometry in KD is novel" — **false / already established**;
- "cosine student-teacher anomaly distillation is novel" — **false / already established**.

A more defensible SABRA contribution, if later validated, is mechanism-specific:

> A scale-invariant directional student can improve correction angle while catastrophically destabilizing anomaly-score calibration when the student's signed residual is injected into logits; a teacher-scale-normalized single residual objective restores radial identifiability without returning to multi-objective distillation.

This novelty statement is **provisional**, not publication-ready. A full Scholar/arXiv/web search is still required before making an exhaustive prior-art claim.

---

## 3. Existing-method overlap

P29 already contained a normalized value-regression term, but combined it with sign and pure-normal penalties:

`L_P29 = L_value + L_sign + L_normal`.

P29R1 implicated mixed-objective conflict / gradient starvation. Therefore P30R1 should not simply restore P29.

The useful distinction for the next candidate is:

- P29: one value term **plus multiple auxiliary constraints**;
- P30: one **self-normalized directional** term, radial scale unidentifiable;
- proposed P30R1: one **teacher-scale-normalized residual regression** term, radial scale identifiable with no auxiliary sign/normal/ranking/feature loss.

---

## 4. Candidate formulations

### Candidate A — normalized direction + frozen global scale anchor

Use P30-style direction and reconstitute student residual with a frozen category-agnostic teacher scale.

Strengths:

- cheap;
- zero learned calibration parameters;
- zero or negligible inference overhead.

Weakness:

P30's failure is heavy-tailed, not a uniform scale shift. One global scalar can correct the mean while leaving sample-level q99 outliers intact. It also risks discarding useful per-sample radial variation.

### Candidate B — direction + bounded norm preservation

Add a bounded radial penalty such as log norm ratio or clipped norm ratio.

Strengths:

- directly constrains amplitude;
- easy to understand.

Weaknesses:

- creates at least two gradient components;
- normally introduces a weighting coefficient;
- risks recreating the mixed-objective problem P29R1 already identified.

### Candidate C — single teacher-scale-normalized residual objective **(recommended)**

Working description: **Teacher-Scale Normalized Residual Distillation (TSNRD)**. This is only a descriptive working name, not a novelty claim.

For each sample, flatten the staged correction to 243 coordinates. Retain the existing frozen correction scale `C` and fixed epsilon `eps`.

Let:

`x_t = t / C`

`x_s = s / C`

and define a detached teacher-only sample radius:

`a_t = sqrt(mean(x_t^2) + eps^2)`.

Then use the **same teacher radius for both** vectors:

`z_t = x_t / a_t`

`z_s = x_s / a_t`

with one robust regression objective:

`L_TSNRD = mean SmoothL1(z_s, z_t)`.

Key differences from P30:

1. **do not divide the student by its own RMS**;
2. **do not exclude zero-teacher vectors**;
3. use no sign loss, normal loss, ranking loss, feature loss, or calibration loss.

Why this is geometrically useful:

If `t = alpha*u` and `s = beta*v`, the teacher-only denominator removes cross-sample teacher-scale domination but leaves `beta/alpha` visible to the loss. Angle and radius are therefore corrected by one vector regression rather than separate competing objectives.

For `t = 0`, `a_t = eps` and the target remains zero, so a nonzero student receives an explicit restoring gradient instead of being dropped from the loss.

Expected cost:

- no extra teacher/model forward;
- no extra inference module;
- cached teacher path unchanged;
- approximately O(243) scalar operations per sample;
- expected train overhead versus P30: negligible and plausibly below 5%, to be measured rather than assumed;
- inference overhead: 0% because this is training-loss-only.

Primary risk:

Very small teacher radius can amplify gradients through `1/a_t`. The existing fixed epsilon prevents division by zero, but gradient magnitude on near-zero targets must be explicitly tested before training.

### Candidate D — fixed post-hoc calibration transform

Apply a frozen affine/temperature/robust-scale transform after directional training.

Strength: cheapest possible intervention.

Weakness: likely treats the symptom. P30 residual q99 already explodes before final probability scoring. Post-hoc calibration could hide score saturation while leaving a pathological student residual distribution.

### Candidate E — feature-level norm consistency

Constrain upstream features in addition to the residual.

Strength: potentially useful if later evidence localizes the pathology upstream.

Weakness: current evidence does not. It adds coupling, implementation complexity, and a less clean causal story.

---

## 5. Speed–performance comparison

| Candidate | Direction | Scale stability | Single objective | Category-agnostic | Train overhead | Inference overhead | Hyperparams | Novelty risk | Recommendation |
|---|---|---|---|---|---|---|---|---|---|
| Normalized direction + frozen global scale | strong | medium | potentially | yes | very low | ~0 | 0–1 frozen scalar | high | secondary |
| Direction + bounded norm term | strong | strong | no / effectively multi-term | yes | low | 0 | usually >=1 weight | high | avoid first |
| **Teacher-scale-normalized residual regression** | **strong expected** | **strong expected** | **yes** | **yes** | **very low** | **0** | **0 new if existing C/eps reused** | medium | **PRIMARY** |
| Fixed calibration transform | unchanged | output-only | yes | yes | ~0 | tiny | 1–2 frozen stats | high | reject as first fix |
| Feature-level norm consistency | uncertain | uncertain | usually no | yes | medium+ | 0 or more | >=1 | high | defer |

---

## 6. Novelty analysis

The strongest novelty path is **not** a generic norm-and-direction claim.

Potential contribution stack:

1. **Mechanism finding:** in SABRA's residual-to-logit deployment, scale-invariant directional distillation improved cosine yet produced catastrophic heavy-tail residuals and normal-score saturation.
2. **Zero-target finding:** excluding exact zero teacher corrections removes a crucial restoring gradient for normal-like source samples.
3. **Method response:** use one teacher-scale-normalized residual regression so direction and radius are jointly identifiable without multiple auxiliary losses.
4. **Engineering property:** source-cache compatible, no new model forward, no inference overhead, no new learned calibration network, and no class-specific scale.
5. **Ablation clarity:** P29 multi-objective vs P30 self-normalized direction-only vs P30R1 teacher-scale-normalized single objective gives a clean causal sequence.

Novelty confidence today: **medium / provisional**. Direct norm+direction KD prior art exists, so publication language must emphasize the residual-to-logit anomaly-calibration mechanism and the single teacher-scale geometry rather than claiming a new generic KD principle.

---

## 7. Ranked candidates

1. **Teacher-scale-normalized single residual regression (Candidate C)** — best speed/performance/science Pareto point.
2. **Normalized direction + frozen global scale anchor (Candidate A)** — very cheap but likely too coarse for the observed q99 heavy tail.
3. **Bounded norm preservation (Candidate B)** — scientifically plausible but risks recreating multi-objective conflict and coefficient tuning.

Feature-level consistency and post-hoc calibration are not justified as the next step.

---

## 8. Primary recommendation

### Recommended formulation

Use **one teacher-scale-normalized robust residual regression objective**:

`a_t = sqrt(mean((t/C)^2) + eps^2)`

`L = SmoothL1((s/C)/a_t, (t/C)/a_t)`

with `a_t` detached and computed per sample from the cached teacher. Include zero-teacher samples. Do not self-normalize the student.

### Why

It directly fixes both mathematical blind spots in P30 while preserving the P29R1 lesson:

- direction remains learnable;
- student radius is no longer free;
- zero teacher produces a zero-restoring gradient;
- large student tails are penalized before they can saturate logits;
- one loss means no auxiliary-gradient conflict;
- no new inference path or class-specific scale is required.

### Why not the alternatives

- Global scale anchor is too coarse for a heavy-tail failure.
- Separate norm penalty needs another gradient term / coefficient.
- Post-hoc calibration can mask rather than repair the pathological residual.
- Feature consistency is unsupported by current evidence and costs causal clarity.

### Expected behavior — hypotheses, not claims

- directional cosine: retain much of P30's gain if teacher vector regression is learnable;
- residual q99: fall sharply relative to P30 because large `beta` is no longer loss-invariant;
- normal-score q99: fall sharply if residual tail is the causal driver, as local deployment analysis indicates;
- pAP: recover substantially from P30 if the AP collapse is predominantly score saturation;
- AUROC: should remain competitive, but no exact numeric improvement is preregistered yet;
- gradient health: finite/nonzero with additional useful gradients on zero-teacher samples;
- runtime: near P30 and substantially simpler than a multi-loss or feature-level method;
- inference: unchanged.

### Main risk

Near-zero teacher radii may over-amplify gradients, or the teacher's radial information itself may be too noisy to transfer. Either outcome should kill the method cheaply before another full one-class schedule.

---

## 9. Cheapest falsification experiment

No scientific P30R1 training should begin yet.

Use this gate order:

### Gate 1 — symbolic / synthetic scale tests

For `s = beta*u`, `t = alpha*u`, test `beta/alpha`:

- 0.1x;
- 1x;
- 10x;
- 100x.

Required behavior:

- unique minimum near 1x;
- monotonic penalty for severe over/under-scale;
- unlike P30, 10x and 100x must **not** remain approximately loss-equivalent to 1x.

Also test:

- opposite direction at correct magnitude;
- correct direction with catastrophic magnitude;
- exactly zero teacher with nonzero student;
- near-zero teacher;
- mixed sample scales.

### Gate 2 — synthetic gradient test

Require:

- finite student gradients;
- no NaN/Inf;
- nonzero restoring gradient for zero-teacher / nonzero-student case;
- no gradient explosion on near-zero targets;
- teacher detached/frozen.

### Gate 3 — source-cache-only diagnostic

Using existing allowed cached source tensors only, inspect:

- distribution of teacher per-sample RMS;
- fraction of exact/near-zero teacher targets;
- student/teacher radial ratio distribution after a tiny smoke;
- student residual q99, especially on zero-teacher source samples.

Do not use held labels to tune any constant.

### Gate 4 — one-step / short training smoke

Verify:

- full cached train → backward → optimizer path;
- student changes, teacher does not;
- no new CLIP/Phase2B forward;
- residual tail does not immediately diverge.

### Gate 5 — 40-step speed profile

Measure against frozen P30/P29. Do not assume overhead.

Target:

- inference overhead = 0%;
- training overhead preferably <10%;
- if >15%, investigate before Stage 2.

Only after all gates pass should a separately preregistered P30R1 one-class Stage 2 be considered.

---

## 10. P30R1 preregistration recommendation

Freeze before implementation/execution:

- exact teacher-scale normalization equation;
- exact epsilon and correction scale source;
- whether SmoothL1 beta is default/fixed;
- inclusion of zero-teacher samples;
- no student self-normalization;
- no auxiliary sign/normal/ranking/feature/calibration terms;
- source-only synthetic and radial-stability gates;
- gradient-norm safety bounds;
- runtime acceptance gate;
- one-class identity chosen before observing P30R1 results;
- Stage 2 AP/AUROC/directional/radial/q99 stop criteria;
- no class-specific scale;
- zero additional inference module;
- rerun/version policy and evidence paths.

The preregistration should explicitly state that **P30R1 is not a P30 rerun**. It is a new hypothesis generated from the terminal P30 Stage 2 result.

---

## Final decision

**Primary recommendation: teacher-scale-normalized single residual regression.**

The next scientific question is no longer "does direction help?" P30 already answered that partially: direction improved. The next question is:

> Can one teacher-scale-normalized vector objective retain directional transfer while making the student's radial correction identifiable enough to prevent heavy-tail logit saturation?

Do not add a separate magnitude loss first. Do not add feature distillation. Do not apply post-hoc calibration as the primary fix. Kill this candidate with synthetic scale/zero-target/gradient tests if it cannot bound radius cheaply.

## Research-access limitation

The repository/local-evidence analysis is direct. Public prior-art checks above were verified through accessible GitHub repositories and their linked paper metadata. NVIDIA AI-Q research was not available because no reachable AI-Q backend was configured in this environment, and broad web search was unavailable. Therefore the novelty assessment is intentionally marked provisional and must not be presented as an exhaustive literature review.