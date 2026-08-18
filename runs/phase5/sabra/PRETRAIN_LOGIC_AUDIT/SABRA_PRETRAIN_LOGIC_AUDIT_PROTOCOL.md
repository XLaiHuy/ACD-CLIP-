# SABRA pre-training logic audit protocol

Status: **FROZEN BEFORE SCIENTIFIC GT EVALUATION**

This is a VisA-only falsification and training-authorization audit. It asks
whether the SABRA architecture logic is sufficiently supported to justify a
separate preregistered full-training study. It does not train SABRA, select a
final model, evaluate MVTec, evaluate medical images, or choose a production
correction magnitude or threshold.

## 1. Provenance and firewall

- Branch: `research/p5-sabra-g`.
- Reconciled setup starting HEAD: `1235b4506da5eead8419bb57eb1ecafcbda9775b`.
- Setup readiness artifact: `runs/phase5/sabra/PRETRAIN_SETUP_AUDIT/READINESS_DECISION.json`, whose terminal is required to be `PRETRAIN_LOGIC_AUDIT_READY`.
- Phase2B checkpoint: `runs/phase4v/v1_7/readiness_full/adapter_5.pth`, SHA-256 `a786e1498254d4589824e7bd111e1917b399d31687ed807212e86dfe3893df34`.
- Phase2B config: `runs/phase4/k1/short64_seed0_attempt5/config.json`, SHA-256 `377ce1c0ae1dd870f82ddcb828d8d8809fa46c007e61567f2150ec11354b23a4`.
- CLIP asset SHA-256: `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`.
- VisA metadata: `dataset/hub/VisA.jsonl`, SHA-256 `468463d2d6234fa7537c6da32b027758527676a12a54a4028c5a282cdd726842`.
- Dataset: VisA only; classes are the 12 fixed classes in `tools/sabra/data.py`.
- The GT-free cache is generated from RGB images and public class/image identity only. No label, mask path, or mask pixel may be read before the cache is finalized and hashed.
- MVTec reads, medical reads, and Phase2B training steps are counters and must each remain zero.
- Phase2B is inference-only, `eval()`, all parameters frozen, H6 disabled, no optimizer, backward, LoRA, adapter, prompt, routing, or feature modification.

The current study becomes `PRETRAIN_LOGIC_AUDIT_INVALID` if readiness, source
identity, checkpoint/config identity, domain firewall, geometry parity, or
Need deployment parity fails; if a scientific definition changes after GT,
the current study is invalidated and a fresh protocol is required.

## 2. Frozen image, feature, and deployment semantics

Images use deterministic resize to 518x518 (bicubic), tensor conversion, and
the frozen CLIP mean/std in `tools/sabra/data.py`. The Phase2B visual output
is the three normalized stage feature arrays with shape `[3,1369,768]` and
native logits `[3,1369,2]` / margins `[3,1369]` from
`ACDCLIP.vision_text_fusion_gate_seg(..., test_mode=True, domain="Industrial", return_details=True)`.

The deployed path is exactly: native logits -> 7x7 Gaussian blur with sigma
1 -> bilinear resize to 518x518 with `align_corners=True` -> arithmetic mean
of the three stages -> two-class softmax. The abnormal probability is the
class-1 output. The source output and sidecar reconstruction must agree within
`1e-6` maximum absolute error on the fixed parity samples.

Native mean margin is the mean of the three stage margins. Its within-image
percentile rank uses all 1369 patches, ascending order, average rank for ties,
and denominator `max(1369-1,1)`. Robust margin is
`(mean_margin - median(mean_margin))/(MAD(mean_margin)+1e-6)`, with no clipping.
For each stage, stage-margin percentile rank uses the same rule. `D_rank` is
the population standard deviation (`ddof=0`) across the three stage percentile
ranks at each patch. No GT enters any of these quantities.

## 3. Frozen B1 top-8 and p9 construction

For each image, the candidate pool is the conjunction of:

1. `D_rank < median(D_rank)` across all 1369 patches;
2. every stage anomaly-margin percentile rank is `< 0.5`; and
3. Chebyshev patch-grid distance from the query is `> 3`.

The search representation is the L2-normalized mean of the three aligned
normalized stage features. Candidates are sorted by descending cosine
similarity to the query representation, with ascending patch index as the
tie-break. The primary peers are positions p1 through p8. p9 is the exact
next valid candidate in this same ordering, never a fabricated or re-ranked
candidate. The same peer IDs are used for all three stages.

`valid_b1` is true only when at least eight candidates exist. `valid_stability`
is true only when p9 exists, hence only when at least nine candidates exist.
Exactly eight candidates gives `valid_b1=true` and `valid_stability=false`.
Fewer than eight gives both false. Invalid patches have no scientific score.

## 4. Frozen compact geometry and relational sensors

For each valid top-8 reference set and stage, persist query-to-peer cosine
`c[8]` and the upper-triangular peer Gram matrix `G[8,8]` in the canonical
36-value `diag8_then_offdiag28` representation. Persist query-to-p9 cosine
and p9-to-top-8 cosine separately. Compact replacement geometry is formed by
replacing only slot j in `c` and the corresponding row/column of `G`, setting
the replacement diagonal to one.

The canonical source modules are `tools/p5f_geometry/pgm.py`,
`tools/p5f_geometry/pcrr.py`, and `tools/p5f_geometry/common.py`; their source
hashes are recorded in the cache manifest. The frozen configurations are:

- **PGM:** `pgm_sum_whitened_mean`: machine-rank centered eigensystem,
  sum-whitened coordinates, mean over the three stage percentile scores.
  With `H=I-11^T/8`, `w=1/8`, `C=HGH`, `b=H(c-Gw)`, eigenvalues/eigenvectors
  ordered descending, and tolerance
  `eps(float32)*max(1,max(lambda))*8`, the stage raw value is the sum over
  positive machine-rank eigenvalues of
  `7*(b dot v_j)^2/lambda_j^2`.
- **PCRR:** `pcrr_witness_local_mean_mean`: per-peer local witness values
  `(1 + count(other peer distances <= query-to-peer distance))/8`, with the
  peer itself excluded; mean the eight witness values after deterministic
  ascending sort, then mean the three stage percentile scores. PCRR is
  optional and cannot be rescued by a new configuration or fusion.

For each family and each image, the baseline CDF for each stage raw component
contains **all valid_b1 patches in that image only**. Average ties are assigned
their mid-percentile. Invalid-reference patches are excluded from the CDF and
receive zero. Baseline scores are mapped through this frozen baseline CDF.
Every p9 replacement score is mapped through the same baseline CDF; a
perturbed image never receives a new CDF or independent reranking.

For PGM and, analogously, PCRR, define baseline evidence `E`, replacement
ranks `r_1,...,r_8`, boundary replacement `r_8`,
`S_boundary = 1-|r_8-r_0|`,
`S_influence = 1-max_j|r_j-r_0|`, and
`R=min(r_0,...,r_8)`. Stability values are clipped to [0,1] only for numeric
under/overflow protection. `R` is worst-case robust evidence, not pure
stability. For PGM, `S_P=min(S_boundary_P,S_influence_P)` and
`T_P=min(R_P,S_P)`. If PCRR is retained, `S_R=min(S_boundary_R,S_influence_R)`,
`T_R=min(R_R,S_R)`, and `TRUST=min(T_P,T_R)`; otherwise `TRUST=T_P`.

## 5. Geometry parity gate

Before any scientific interpretation, verify that compact top-8 c/G produces
the same canonical PGM and PCRR output as the frozen implementation; that all
eight p9 replacement results from compact statistics match direct feature-space
construction on a fixed predeclared sample; and that any accelerated
eigendecomposition agrees with the NumPy float64 reference. Record sample
identities, dtype, tolerance, maximum absolute error, and relative error.
No formula may be changed to make parity pass. Failure invalidates the study.

## 6. GT-free cache freeze

One immutable cache is written before mask pixels are opened. Each record is
identified by class, image path identity, and patch index and contains native
logits/margins, mean margin, margin rank, robust margin, stage ranks, `D_rank`,
deployed score/margin if needed, `peer_indices[1369,8]`, p9 index,
`valid_b1`, `valid_stability`, candidate count, B1 centroid evidence, compact
query-peer cosines, compact peer Gram, query-p9 cosines, p9-to-peer cosines,
baseline PGM/PCRR, eight fixed-K replacement PGM/PCRR scores, p8/p9 gap, PGM
eigensystem diagnostics, PCRR tie/comparison diagnostics, and deployment
sensitivity. Raw RGB and persistent full 768-D features are forbidden.

The cache shards, aggregate manifest, protocol, source commit, checkpoint,
config, metadata, and implementation hashes are recorded in
`GT_FREE_CACHE_MANIFEST.json`. The manifest must say
`GT_FREE_CACHE_FINALIZED=true`. After finalization the cache is immutable;
any change invalidates the study. Implementation and cache-manifest metadata
are committed before scientific GT evaluation.

## 7. Scientific targets and module diagnostics

After cache finalization, masks are resized to 518x518 with nearest-neighbor.
Patch occupancy is the mean of each non-overlapping 14x14 block in the 37x37
grid. Binary patch anomaly GT is `occupancy > 0`; continuous occupancy is used
only for Spearman diagnostics.

### PGM

PGM is tested as a relational anomaly sensor, not as a Need predictor. Report
per class AUROC, AP, prevalence-aware normalized AP when defined, Spearman
correlation with occupancy, B1 comparison, effect versus 0.5, class support,
and negative tail. A practically positive class has delta >= 0.01; neutral is
(-0.01,0.01); practically negative is <= -0.01. ROPE is frozen to
[-0.01,+0.01]. A delta <= -0.05 is a negative-tail warning. At least two such
classes or median delta < -0.01 is catastrophic. `PGM_STATUS` is SUPPORTED,
PROMISING_BUT_UNCERTAIN, INCONCLUSIVE, or FALSIFIED according to practical
effect, class distribution, uncertainty, and safety tail; 12/12 wins and
p<0.05 are not required. Default promising directional support is at least
8/12 non-neutral positive classes; strong support is at least 9/12 plus strong
mean/median evidence.

### PCRR

PCRR is an optional incremental diagnostic. In deterministic leave-one-class-
out folds, fit fixed low-capacity logistic Model A on PGM rank and Model B on
PGM rank plus PCRR rank, using only training classes. Primary effect is
held-out class AUROC(B)-AUROC(A). RETAIN requires positive mean, non-negative
median, reasonably consistent class support, and no catastrophic tail;
otherwise PCRR_STATUS is DROP and SABRA is not failed by PCRR alone.

### Trust

Trust is tested as relational-evidence correctness at controlled baseline
evidence strength. In deterministic leave-one-class-out folds, fit the same
fixed logistic diagnostic with mandatory E covariate for each of: E alone,
E+S_boundary, E+S_influence, E+R, and E+T. Report class-level AUROC and
incremental deltas against E, including a within-image E-quartile stratified
diagnostic. The primary Trust effect is AUROC(E+T)-AUROC(E). This does not
interpret stability alone as correctness. Stable-but-wrong and reference
contamination audits below are mandatory and can veto Trust support.

### Need oracle and parity

Need asks whether the exact positive intervention improves the final deployed
loss. For a scalar `delta_i`, add the same infinitesimal positive abnormal-logit
correction at native patch i across all three stages. With frozen Phase2B,
`u_signed_i = -dL/d(delta_i)`, where L is the exact deployed segmentation loss
(`utils.calculate_seg_loss`) after the exact blur/resize/stage-mean/softmax
path. Persist `u_positive=max(u_signed,0)`, `u_harm=max(-u_signed,0)`, and the
binary diagnostic `u_signed > 1e-8`. No finite correction magnitude or
production threshold is selected.

Finite-difference parity uses epsilon `1e-4`, the fixed predeclared sample
identities and patch indices `[0,684,1368]`, sign tolerance `1e-8`, absolute
tolerance `2e-3`, and relative tolerance `2e-2`. Sign parity is required for
all non-tolerance cases at rate >= 0.999 and numerical agreement must satisfy
`abs(FD-u) <= 2e-3 + 2e-2*max(abs(FD),abs(u))`. Meaningful unexplained sign
disagreement invalidates the study.

Need features are exactly: mean-margin within-image percentile rank,
robust-normalized mean margin, `D_rank`, and mean absolute deployment
sensitivity of final abnormal probability to the shared native patch delta.
Need cannot use RGB, full features, class/dataset ID, GT, PGM, PCRR, or Trust.

Need diagnostics are deterministic 12-fold leave-one-VisA-class-out, with all
preprocessing fit only on the 11 training classes. C1 is balanced logistic
regression on the four fixed features. C2 is balanced logistic regression on
each feature's standardized value plus three positive hinge terms at training
fold q25/q50/q75, with no interactions. Both use solver `lbfgs`, `C=1.0`,
`max_iter=1000`, `random_state=0`; no sweep or rescue. Required baselines are
margin-only, D_rank-only, margin+D_rank, C1, and C2. Primary Need effect is
the best preregistered valid capacity versus margin-only on held-out
`utility_positive`; signed utility Spearman, Brier/calibration, harm ranking,
and per-class deltas are also required. C1 is the minimum capacity if valid;
otherwise C2 may be selected only if it passes. Failure of both blocks
authorization.

### Authority

Only out-of-fold Need predictions are used. Let N be the selected OOF Need
score, E baseline PGM evidence, and T structural Trust. Compare N, N*E, and
N*T for utility-positive AUROC, signed utility, harm rate, top-ranked utility,
risk-coverage, and coverage. The primary Authority effect is
AUROC(N*T)-AUROC(N*E); also report `AUROC(N*T)-AUROC(N)` and the fixed
Phase2B margin/confidence comparison. If N*E consistently matches or beats
N*T, Trust authority is not justified. No hard threshold or finite correction
is selected.

## 8. Reference credibility and coverage

Using GT only after cache freeze, report occupancy of p1-p8 and p9, fraction
of reference sets containing at least one anomalous peer, fraction containing
at least two, clean-versus-contaminated behavior, and high-/low-Trust false
evidence rates. A `STABLE_BUT_WRONG` row is a valid patch with `E>=0.75`,
`T_P>=0.75`, and binary patch GT negative; all such rows are written to
`STABLE_BUT_WRONG.csv` with class/image/patch, peer IDs, occupancies, E,
stability, robust evidence, and Trust. High-Trust systematic contamination
or a catastrophic negative tail vetoes Trust support.

Stability coverage is `valid_stability/valid_b1`, overall and by class.
Strong coverage is overall >=90% and every class >=75%; acceptable is overall
>=80%, no more than two classes <70%, and no class <50%; otherwise coverage is
insufficient and Trust cannot be SUPPORTED/PROMISING. p9 is never invented.

## 9. Statistics, evidence levels, and authorization

The statistical unit is VisA class (N=12), never a pixel or patch. For every
class-level effect report mean, median, 10,000-repetition class bootstrap 95%
CI, supportive/neutral/negative class counts, worst classes, raw exact paired
sign-flip p-value, and Holm-adjusted p-value. The exact paired sign-flip uses
all `2^12=4096` sign assignments. Holm adjustment covers the four core
hypotheses H_PGM, H_TRUST, H_NEED, and H_AUTHORITY; PCRR is reported
separately. Raw and adjusted p-values are evidence annotations, not the sole
authorization gate.

Scientific statuses are SUPPORTED, PROMISING_BUT_UNCERTAIN, INCONCLUSIVE, or
FALSIFIED. Implementation/provenance gates are PASS or FAIL. Full training is
authorized only if all integrity/parity/firewall gates pass, PGM, Trust, Need,
and Authority are each SUPPORTED or PROMISING_BUT_UNCERTAIN, none is
INCONCLUSIVE/FALSIFIED, and no catastrophic safety/reference failure exists.
All four SUPPORTED gives `AUTHORIZATION_STRENGTH=STRONG`; otherwise it is
`PROVISIONAL`. PCRR DROP does not block authorization; the authorized
relational set is then PGM.

The final decision must include all required artifact paths, provenance and
integrity flags, p9 coverage, component statuses/effects/p-values/tails,
stable-but-wrong summary, counters, failed or inconclusive core hypotheses,
next action, exact terminal status, and the literal answer
`FULL_SABRA_TRAIN_AUTHORIZED=true|false`.

## 10. Adversarial invalidation checklist

Before result certification, explicitly audit GT leakage, deployment mismatch,
compact/direct geometry mismatch, pseudo-replication, class dominance,
evidence-strength confounding, stable-but-wrong references, p9 identity,
perturbation reranking, Need forbidden inputs, OOF violations, multiplicity,
historical MVTec contamination, practical/statistical contradiction, and
uncertainty/falsification contradiction. A result-changing defect makes the
study invalid; scientific failures are recorded and are not rescued by new
configurations, capacities, thresholds, MVTec, medical evaluation, or
training.
