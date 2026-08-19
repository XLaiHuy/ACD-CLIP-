You are continuing the Phase5 SABRA research program in repository:

    /workspace/ACD-CLIP-

You are responsible for executing a NEW FOLLOW-UP STUDY:

    SABRA Trust-v2 / Reference Credibility Development
    followed, only after a full freeze, by external industrial validation.

This prompt is the scientific and engineering contract for the study.
Do not silently reinterpret, simplify, widen, or redesign it.

======================================================================
A. SCIENTIFIC CONTEXT — SOURCE OF TRUTH
======================================================================

The previous SABRA pre-training logic audit is COMPLETE and MUST NOT be
rewritten, reinterpreted, rerun merely to obtain a better result, or treated
as invalid.

Previous terminal:

    FULL_SABRA_TRAIN_NOT_AUTHORIZED

Previous component conclusions:

    PGM       = SUPPORTED
    Need      = SUPPORTED
    Trust     = INCONCLUSIVE
    Authority = INCONCLUSIVE
    PCRR      = DROP

Previous final handoff commit known from the completed study:

    2c895219e4f0b4f74e733cffb684d67e70643b89

Previous repaired result commit:

    63ec4be75b2228eff536e79fc81af05bf83360b4

Previous old result SHA before LFS repair:

    f2bef23f66e9f113685ad9ef6beab8e0edaec0c5

The LFS repair changed storage representation only.
Scientific content was unchanged.

IMPORTANT SCIENTIFIC INTERPRETATION:

1. PGM is now frozen as the primary relational anomaly evidence sensor.
2. Need C1 is now frozen as the minimum supported Need capacity.
3. PCRR must NOT be rescued as a primary anomaly sensor by increasing weight,
   tuning fusion coefficients, adding learned fusion, or sweeping variants.
4. Current Trust based mainly on p9 perturbation stability/robustness did not
   provide practically useful incremental information over PGM evidence E.
5. The new hypothesis is therefore:

       false PGM evidence may be better predicted by
       REFERENCE CREDIBILITY / PEER-CLOUD QUALITY

   rather than by single-neighbor p9 stability alone.

6. This is a FOLLOW-UP DEVELOPMENT STUDY informed by the previous VisA
   results.

   Therefore:
   - DO NOT describe VisA as unseen/blinded confirmation.
   - VisA is development/falsification data for Trust-v2.
   - MVTec may be used only AFTER the final Trust-v2 candidate is completely
     frozen and committed/pushed.
   - MVTec is the external industrial validation.
   - Medical data must remain completely untouched.

======================================================================
B. RESULT PHILOSOPHY — DO NOT FORCE PERFECT NUMBERS
======================================================================

Do NOT force all metrics, classes, confidence intervals, or p-values to be
perfect.

Do NOT require:
- 12/12 class wins,
- every CI strictly above zero,
- p < 0.05 for every hypothesis,
- every metric to improve,
- a large AUROC gain if safety improves meaningfully.

Do NOT change gates after seeing results.

This study explicitly allows:

    SUPPORTED
    PROMISING_BUT_UNCERTAIN
    WEAK_POSITIVE_EVIDENCE
    INCONCLUSIVE
    FALSIFIED

This ladder exists because useful scientific evidence can be directional,
consistent, and safety-relevant without being perfect.

However:

    "weak positive evidence"

must NOT be manufactured by shrinking a threshold after seeing the results.

All interpretation rules below must be written into the protocol and pushed
BEFORE Trust-v2 scientific evaluation.

The previous study remains judged by its previous frozen protocol.
The new evidence ladder applies ONLY to this new Trust-v2 study.

======================================================================
C. HARD DOMAIN FIREWALL
======================================================================

During Trust-v2 DEVELOPMENT:

ALLOWED:
    VisA RGB
    frozen Phase2B inference
    previous GT-free artifacts
    new GT-free Trust-v2 feature cache
    VisA GT only after the new feature cache is frozen
    low-capacity sklearn diagnostic models

FORBIDDEN:
    MVTec during feature design/model selection
    medical images
    medical metadata
    medical masks
    medical directory traversal
    medical file hashing that opens file content
    any Phase2B training
    any prompt training
    any LoRA training
    any adapter training
    H6 revival
    any feature modification inside Phase2B
    any GT-informed peer selection

After the Trust-v2 candidate is completely frozen and pushed:

ALLOWED:
    one external MVTec evaluation under frozen semantics

FORBIDDEN EVEN THEN:
    MVTec retraining
    MVTec threshold tuning
    MVTec feature selection
    MVTec model selection
    MVTec coefficient fitting
    MVTec hyperparameter tuning
    medical access

Maintain explicit counters:

    MEDICAL_READS
    MVTEC_READS_BEFORE_FREEZE
    PHASE2B_TRAINING_STEPS
    TRUST_V2_MODEL_SELECTION_AFTER_MVTEC

They must all remain zero except normal MVTec reads AFTER freeze.

======================================================================
D. GIT / REMOTE RESUME SAFETY
======================================================================

Operate on:

    research/p5-sabra-g

Do NOT create a replacement branch unless a concrete repository blocker makes
the current branch unusable.

At startup:

1. cd /workspace/ACD-CLIP-
2. git status --short
3. git branch --show-current
4. git fetch origin research/p5-sabra-g
5. record:
       local HEAD
       remote HEAD
       divergence
6. verify commit
       2c895219e4f0b4f74e733cffb684d67e70643b89
   is an ancestor of the working branch.

If local is clean and behind only:
    fast-forward only.

If remote has advanced while local has overlapping work:
    do NOT rebase/reset/stash/force.
    stop with:

        REMOTE_ADVANCED_DURING_TRUST_V2

and report:
    local SHA
    remote SHA
    divergence
    overlapping paths

Never:
    git reset
    git rebase
    git stash
    git clean
    git push --force
    git push --force-with-lease
    git add .
    git commit --amend on already-pushed history

Exact-path staging only.

Normal fast-forward commits and pushes required by this prompt are already
explicitly authorized.
Do NOT ask permission again for ordinary commits/pushes.

Before EVERY commit boundary:
    git fetch origin research/p5-sabra-g
    verify remote-race safety.

======================================================================
E. FROZEN PHASE2B PROVENANCE
======================================================================

Use the exact previous assets.

Checkpoint:

    runs/phase4v/v1_7/readiness_full/adapter_5.pth

SHA-256:

    a786e1498254d4589824e7bd111e1917b399d31687ed807212e86dfe3893df34

Config:

    runs/phase4/k1/short64_seed0_attempt5/config.json

SHA-256:

    377ce1c0ae1dd870f82ddcb828d8d8809fa46c007e61567f2150ec11354b23a4

CLIP:

    /workspace/ACD-CLIP-/.runtime/assets/ViT-L-14-336px.pt

SHA-256:

    3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02

VisA metadata SHA-256:

    468463d2d6234fa7537c6da32b027758527676a12a54a4028c5a282cdd726842

Environment:

    conda env torchhuy
    Python 3.10
    torch 2.5.1+cu121 expected unless verified-compatible runtime differs

Environment variables:

    ACDCLIP_DATA_ROOT=/workspace/data
    ACDCLIP_CLIP_VITL14_336=/workspace/ACD-CLIP-/.runtime/assets/ViT-L-14-336px.pt

Phase2B must remain:

    eval()
    frozen
    inference-only
    H6 disabled

No optimizer.
No backward through model parameters.
No Phase2B weight mutation.

Sidecar implementation only.
Do NOT modify model/adapter.py unless a concrete parity defect proves it is
absolutely required; if that occurs, STOP instead of casually editing it.

======================================================================
F. EXISTING FROZEN SEMANTICS TO REUSE EXACTLY
======================================================================

Visual stages:

    [8, 16, 24]

Aligned features:

    [3, 1369, 768]

Native logits:

    [3, 1369, 2]

Native margin per stage:

    anomaly_logit - normal_logit

D_rank:

1. compute stage margin percentile rank over all 1369 patches in the image,
   ascending,
   average rank for ties,
   denominator 1368.

2. per patch:

       D_rank = population_std([rank8, rank16, rank24])

   ddof = 0.

B1 candidate pool for query q:

    D_rank(candidate) < image median D_rank

AND

    stage_rank8  < 0.5
    stage_rank16 < 0.5
    stage_rank24 < 0.5

AND

    Chebyshev patch-grid distance(query, candidate) > 3

Search representation:

    shared = L2_normalize(
        mean(stage8_feature, stage16_feature, stage24_feature)
    )

Ordering:

    descending cosine(query_shared, candidate_shared)

Tie break:

    ascending candidate patch index

Primary peers:

    p1 ... p8

Local reserve:

    p9 = exact ninth candidate in the SAME ordering.

Do not rerank.
Do not invent p9.

Same peer IDs are used for all three stages.

PGM remains canonical:

    pgm_sum_whitened_mean

PCRR remains canonical:

    pcrr_witness_local_mean_mean

Do NOT modify PGM or PCRR formulas in this study.

======================================================================
G. NEW TRUST-V2 CORE HYPOTHESIS
======================================================================

Current p9 stability asks roughly:

    "does evidence collapse when one close reserve peer replaces one selected
     peer?"

That is insufficient when:

    peer cloud is internally coherent
    BUT
    peer cloud is semantically unsuitable as a reference for the query.

Trust-v2 therefore asks:

    "Given baseline PGM evidence E, does the structure and credibility of the
     peer/reference cloud explain whether that evidence is correct?"

This is a CONDITIONAL question.

Trust-v2 is NOT allowed to simply become a stronger anomaly detector by
smuggling in RGB/full CLIP features/class identity.

======================================================================
H. NEW MEDIUM RESERVE p16 — EXACT DEFINITION
======================================================================

Keep p9.

Add exactly one additional reserve:

    p16 = exact candidate at rank 16

because:

    K = 8
    K+1 = 9  -> local boundary reserve
    2K  = 16 -> medium neighborhood reserve

Do NOT add p32 in this study.
Do NOT sweep reserve ranks.
Do NOT choose p16 based on results.

p16 must use:

    exact same B1 candidate pool
    exact same shared representation
    exact same ordering
    exact same tie-break

No independent reranking.

Validity:

    valid_p9  iff >= 9 candidates
    valid_p16 iff >= 16 candidates

Never fabricate missing reserves.

For p16 stress tests:
    always keep K=8.

For every j in 1..8:

    replace p_j with p16

and preserve the other seven peer IDs.

Do not add a ninth peer.
Do not change K.

======================================================================
I. IMPORTANT STABILITY SEMANTIC CLEANUP
======================================================================

The previous definitions include:

    S_boundary = 1 - |r8 - r0|

and:

    S_influence = 1 - max_j |rj - r0|

where j includes j=8.

Therefore mathematically:

    S_influence <= S_boundary

and consequently:

    min(S_boundary, S_influence) == S_influence

up to numerical precision.

This is NOT a reason to invalidate the old study.

But in Trust-v2:

- retain S_boundary only as a descriptive diagnostic;
- do NOT treat S_boundary as an independent predictive feature;
- use worst single-peer influence stability as the actual stability feature.

For reserve rho in {p9, p16}:

    E = baseline PGM evidence rank

    r_rho_j = PGM evidence after replacing p_j by reserve rho

    S_rho =
        1 - max_j |r_rho_j - E|

    R_rho =
        min(E, r_rho_1, ..., r_rho_8)

Clip S only to [0,1] for numerical safety.

Perturbed scores MUST be mapped through the SAME frozen baseline image CDF.

Never construct a new perturbation CDF.
Never rerank perturbations independently.

======================================================================
J. TRUST-V2 REFERENCE-CREDIBILITY FEATURES
======================================================================

The feature set is FIXED here.

No new feature may be added after VisA results are inspected.

All features are GT-free.

--------------------------------------------------
J1. peer_coherence
--------------------------------------------------

For each stage g:

    G_g = 8x8 peer Gram matrix.

Take the 28 off-diagonal upper-triangle cosine similarities.

    coherence_g =
        mean_{i<j} G_g[i,j]

Aggregate:

    peer_coherence =
        mean(coherence_8, coherence_16, coherence_24)

Interpretation:

    high -> peers mutually form a coherent feature neighborhood
    low  -> peer cloud is heterogeneous

Do not threshold it manually.

--------------------------------------------------
J2. query_support_mean
--------------------------------------------------

Using the shared search representation:

    s_i = cosine(query_shared, p_i_shared), i=1..8

Define:

    query_support_mean = mean(s_1 ... s_8)

This asks whether the query has a genuinely dense set of similar candidates
or whether top-8 selection is being forced from weak support.

--------------------------------------------------
J3. peer_eigen_entropy
--------------------------------------------------

For each stage:

    H = I - 11^T / 8
    C = H G H

Use the SAME machine-rank positive-eigenvalue tolerance as canonical PGM.

Let positive eigenvalues be:

    lambda_1 ... lambda_m

If no positive eigenvalue:
    entropy_g = 0

Otherwise:

    p_j = lambda_j / sum(lambda)

    entropy_g =
        -sum_j p_j * log(p_j) / log(7)

Use natural log.

Clamp only numerical drift into [0,1].

Aggregate:

    peer_eigen_entropy =
        mean(entropy_8, entropy_16, entropy_24)

Do NOT also add effective-rank as another model feature because it is
monotonic/redundant with this entropy in this fixed study.

--------------------------------------------------
J4. stage_query_profile_disagreement
--------------------------------------------------

For each stage g:

    c_g[8] = query-to-peer cosine vector

For each stage pair (g,h):

    cos_profile(g,h) =
        dot(c_g,c_h) /
        ((||c_g|| + eps) * (||c_h|| + eps))

eps = 1e-12 in float64 reference arithmetic.

Define:

    stage_query_profile_disagreement =
        mean_{g<h}(1 - cos_profile(g,h))

This measures whether stage 8/16/24 agree on which peers relate strongly to
the query.

Do not convert to a hand-coded good/bad threshold.

--------------------------------------------------
J5. p9 robustness
--------------------------------------------------

Persist:

    S9
    R9

Boundary p8->p9 may be reported but is not a separate model feature.

--------------------------------------------------
J6. p16 robustness
--------------------------------------------------

Persist:

    S16
    R16

Only valid where valid_p16=true.

Do not impute a fake p16 score.

--------------------------------------------------
J7. PCRR disagreement — OPTIONAL DIAGNOSTIC ONLY
--------------------------------------------------

PCRR is NOT revived as anomaly evidence.

Define:

    D_rel =
        abs(PGM_baseline_rank - PCRR_baseline_rank)

Its ONLY question is:

    "does disagreement between two relational sensors predict that PGM
     evidence is unreliable?"

D_rel may enter only the final optional nested model described below.

If it adds no meaningful incremental information:
    PCRR remains DROP.

Never:
    PGM + alpha * PCRR
    weighted fusion
    PCRR threshold sweep
    train a PCRR rescue model

======================================================================
K. GT-FREE TRUST-V2 FEATURE CACHE
======================================================================

Do NOT modify the old immutable SABRA cache.

Create a NEW sidecar cache under:

    runs/phase5/sabra/TRUST_V2_DEVELOPMENT/

Recommended compact artifact:

    TRUST_V2_GT_FREE_FEATURE_CACHE.npz

or sharded NPZ files if required for memory.

Do NOT persist:
    RGB tensors
    full 768-D features
    raw medical/MVTec data

Persist only the compact data needed for reproducibility and science:

identity:
    class
    image identity / deterministic record ID
    patch index

reference validity:
    candidate_count
    valid_b1
    valid_p9
    valid_p16

baseline:
    PGM E
    PCRR baseline rank
    D_rel

credibility:
    peer_coherence
    query_support_mean
    peer_eigen_entropy
    stage_query_profile_disagreement
    S9
    R9
    S16 where valid
    R16 where valid

optional descriptive:
    p8-p9 similarity gap
    p8-p16 similarity gap
    S_boundary9
    S_boundary16

Do not include GT labels or mask occupancy in the GT-free cache.

Write:

    TRUST_V2_GT_FREE_MANIFEST.json

including:
    source commit
    source file hashes
    checkpoint hash
    config hash
    CLIP hash
    VisA metadata hash
    cache shard hashes
    record count
    feature names/order
    dtype
    formulas version
    counters
    GT_FREE_FEATURE_CACHE_FINALIZED=true

Once finalized:
    immutable.

======================================================================
L. REQUIRED CROSS-CHECK AGAINST OLD CACHE
======================================================================

Because B1/PGM/p9 semantics already existed, use the old finalized cache as a
parity anchor.

On a fixed deterministic sample covering:
    >= 3 classes
    >= 5 images/class
    fixed patch indices including:
        0
        684
        1368
        plus deterministic valid-B1 patches

Compare new vs old:

    D_rank
    B1 peer IDs p1..p8
    p9
    baseline PGM
    baseline PCRR
    p9 replacement PGM scores
    S9
    R9

Use justified floating tolerance.

Suggested:
    abs <= 1e-6 for ranks/simple scores
    direct geometry tolerance based on existing canonical parity audit

Do NOT loosen tolerance merely to pass.

If semantic mismatch exists:
    stop with TRUST_V2_BASELINE_PARITY_FAIL

Do NOT run VisA Trust-v2 science.

======================================================================
M. p16 DIRECT / COMPACT PARITY TEST
======================================================================

For a fixed deterministic predeclared sample:

construct p16 replacement geometry two ways:

1. directly from transient frozen Phase2B features;
2. from compact c/G/reserve statistics.

For all 8 replacements:

    compare PGM raw stage outputs
    mapped stage ranks
    final mean PGM E

Also compare PCRR if D_rel pipeline reuses related compact stats.

Record:
    dtype
    max absolute error
    max relative error
    sample IDs
    patch IDs
    reserve IDs

Write:

    P16_GEOMETRY_PARITY_AUDIT.json

Failure:
    invalidates Trust-v2 implementation.
Do not patch formulas after GT.

======================================================================
N. p16 COVERAGE GATE
======================================================================

Report:

    valid_p16 / valid_b1

overall and per VisA class.

Interpretation:

STRONG:
    overall >= 0.90
    every class >= 0.75

ACCEPTABLE:
    overall >= 0.80
    no class < 0.50
    <= 2 classes below 0.70

INSUFFICIENT:
    otherwise

If p16 coverage is INSUFFICIENT:

    p16 model M3 is ineligible.

Do NOT loosen the candidate rule.
Do NOT invent p16.
Do NOT lower rank 16 after seeing coverage.

The p9/base-quality path remains valid.

======================================================================
O. TEST SUITE BEFORE VISUAL GT SCIENCE
======================================================================

Create deterministic tests under a suitable sidecar test path.

Required tests:

1. D_rank old/new parity.
2. B1 candidate and ordering parity.
3. deterministic tie ordering.
4. exact p9 = rank 9.
5. exact p16 = rank 16.
6. p16 is never fabricated.
7. all replacements preserve K=8.
8. no duplicate peer IDs after replacement.
9. compact/direct PGM parity.
10. baseline CDF reused for p9/p16 perturbations.
11. perturbations never build a new CDF.
12. peer_coherence synthetic test.
13. eigen entropy:
      rank-one-like geometry -> low entropy
      balanced multi-direction geometry -> higher entropy
14. stage disagreement:
      identical c profiles -> approximately 0
15. S9/R9 exact synthetic examples.
16. S16/R16 exact synthetic examples.
17. demonstrate:
      min(S_boundary,S_influence)
      equals S_influence within tolerance when influence includes j=8.
18. no GT/mask access in cache builder.
19. no MVTec access in development cache builder.
20. no medical access.
21. no model parameter mutation.
22. deterministic cache reproduction on fixed sample.
23. old-cache cross-parity.

Run tests ONCE unless a concrete implementation defect requires correction.

Do not repeatedly rerun successful tests without reason.

======================================================================
P. NEW PROTOCOL MUST BE FROZEN BEFORE SCIENCE
======================================================================

Before opening VisA masks for Trust-v2 analysis, create:

    runs/phase5/sabra/TRUST_V2_DEVELOPMENT/
        SABRA_TRUST_V2_PROTOCOL.md
        SABRA_TRUST_V2_PROTOCOL.json

The protocol must contain EVERY scientific definition in this prompt,
including:

    feature list
    nested models
    OOF split
    preprocessing
    evidence ladder
    p16 coverage gate
    PCRR role
    safety metrics
    external freeze
    MVTec firewall
    Git provenance

Commit:

    phase5: freeze SABRA Trust-v2 development protocol

Push normally.

Verify:
    local == remote
    divergence 0 0

This protocol commit MUST precede VisA Trust-v2 result generation.

======================================================================
Q. IMPLEMENTATION COMMIT
======================================================================

Implement sidecar code only.

Suggested structure:

    tools/sabra/trust_v2/
        __init__.py
        cache.py
        features.py
        models.py
        science.py
        external.py
        audit.py
        handoff.py

Do not redesign existing PGM/PCRR canonical modules.

After:
    tests PASS
    baseline parity PASS
    p16 parity PASS
    GT-free feature cache FINALIZED
    manifest hash-checked

commit exact relevant paths with message:

    phase5: implement SABRA Trust-v2 audit

Push normally.

Verify remote synchronization before proceeding to VisA scientific analysis.

======================================================================
R. VIS-A SCIENTIFIC TARGET
======================================================================

After protocol + implementation/cache commits are pushed:

VisA mask GT may be used for DEVELOPMENT analysis.

Patch anomaly target:

    mask resized 518x518 nearest-neighbor
    occupancy = mean of each non-overlapping 14x14 patch block
    binary anomaly = occupancy > 0

This target is used for Trust-v2 anomaly-evidence correctness.

Trust-v2 DOES NOT use utility-positive GT as its training target.

Need / Authority use their original utility target separately.

======================================================================
S. TRUST-V2 NESTED MODEL FAMILY
======================================================================

Use deterministic leave-one-VisA-class-out:

    12 folds

Preprocessing:
    training classes only.

Use StandardScaler fitted only on training classes.

All logistic models:

    sklearn LogisticRegression
    class_weight="balanced"
    solver="lbfgs"
    C=1.0
    max_iter=1000
    random_state=0

No hyperparameter sweep.

Models are FIXED nested models:

M0:
    E only

M1:
    E
    peer_coherence
    query_support_mean
    peer_eigen_entropy
    stage_query_profile_disagreement

M2:
    all M1 features
    S9
    R9

M3:
    all M2 features
    S16
    R16

M3 eligible only if p16 coverage gate >= ACCEPTABLE.

M4:
    selected best non-PCRR nested model
    + D_rel

M4 is PCRR incremental diagnostic only.

No arbitrary feature-subset search.
No backward elimination.
No coefficient pruning after results.
No MLP.
No tree ensemble.
No SVM sweep.
No neural network.
No interactions.

======================================================================
T. TRUST-V2 PRIMARY COMPARISON
======================================================================

For each nested model, obtain strictly out-of-fold predictions.

Primary effect:

    AUROC(Mk) - AUROC(M0)

per held-out class.

Also report:
    AP
    normalized AP when valid
    calibration/Brier
    occupancy Spearman
    high-evidence false-evidence rate
    stable-but-wrong rate
    per-class deltas
    mean
    median
    10,000 class bootstrap CI
    exact class-level sign-flip p where applicable

Statistics unit:
    VisA class.

Never patch/pixel as independent statistical units.

======================================================================
U. NEW TRUST-V2 DEVELOPMENT EVIDENCE LADDER
======================================================================

This is a NEW development ladder.
It does NOT reinterpret the previous SABRA study.

Trust-v2 practical AUROC ROPE for this follow-up:

    [-0.005, +0.005]

This is intentionally more permissive than the old study because:

1. Trust is an incremental conditional reliability module;
2. a smaller discrimination gain may still be valuable if intervention harm
   decreases;
3. external MVTec confirmation will still be required before full-training
   authorization.

Per-class catastrophic warning:

    delta AUROC <= -0.03

Catastrophic tail:

    >= 2 classes <= -0.03
    OR
    median delta <= -0.005

Classification:

SUPPORTED:
    mean delta >= +0.010
    median delta >= +0.005
    >= 8/12 classes have delta > 0
    no catastrophic tail

PROMISING_BUT_UNCERTAIN:
    mean delta >= +0.005
    median delta >= 0
    >= 7/12 classes have delta > 0
    no catastrophic tail

WEAK_POSITIVE_EVIDENCE:
    mean delta > 0
    median delta >= 0
    >= 7/12 classes have delta >= 0
    no catastrophic tail

INCONCLUSIVE:
    effect is near zero, mixed, or cannot be distinguished directionally,
    without clear falsification.

FALSIFIED:
    mean or median materially negative
    OR catastrophic tail
    OR systematic degradation.

CI crossing zero does NOT automatically convert PROMISING into INCONCLUSIVE.

p > .05 does NOT automatically mean no evidence.

p < .05 with tiny effect does NOT automatically mean useful.

======================================================================
V. NESTED MODEL SELECTION RULE
======================================================================

VisA is development data.

Select ONE Trust-v2 candidate deterministically:

1. determine evidence class for M1, M2, M3 against M0.
2. among eligible models:
      choose the SIMPLEST model reaching the highest evidence category.
3. if multiple models in same category:
      prefer simpler model unless the more complex model has:
          >= +0.003 additional mean AUROC
          OR a meaningful safety improvement later in Authority.
4. M3 cannot be selected if p16 coverage insufficient.
5. no model below WEAK_POSITIVE_EVIDENCE may be frozen for external
   confirmation as the primary Trust-v2 candidate.

PCRR M4:
    compare selected non-PCRR candidate vs candidate + D_rel.

Retain D_rel only if:
    mean incremental AUROC > 0
    median >= 0
    no catastrophic tail
    and at least WEAK_POSITIVE_EVIDENCE.

Otherwise:
    PCRR remains DROP.

Do not keep PCRR because it is "interesting".

======================================================================
W. TRUST-V2 OUTPUT SEMANTICS
======================================================================

For fairness of downstream Authority:

Generate two OOF conditional probabilities:

    E_cal =
        OOF probability from M0(E-only)

    T_v2 =
        OOF probability from selected Trust-v2 model

Thus:

    E_cal and T_v2 are on the same learned probability scale.

Primary Trust-v2 question remains:

    does T_v2 improve over E_cal?

Do NOT multiply raw PGM by a hand-designed C_ref coefficient and then tune it.

For interpretability, report standardized model coefficients, but do not
interpret coefficient magnitude as causal truth.

======================================================================
X. NEED — FREEZE THE PREVIOUS SUPPORTED DESIGN
======================================================================

Need does NOT get redesigned in this study.

Use the previous supported C1 design:

inputs exactly:
    mean-margin within-image percentile
    robust normalized mean margin
    D_rank
    deployment sensitivity

No:
    RGB
    raw/full CLIP
    PGM
    PCRR
    Trust
    class ID
    dataset ID

Use exact previous LOCO semantics.

For Authority development:
    generate OOF C1 Need score N.

Need target remains:

    utility_positive =
        u_signed > 1e-8

where:

    u_signed = -dL/d(delta_i)

under exact deployed Phase2B semantics.

Do not select finite delta.

Do not modify Need because C2 happened to have a slightly higher mean.
C1 remains the frozen minimum supported capacity.

======================================================================
Y. AUTHORITY-V2
======================================================================

The old Authority failed mainly because current T added almost nothing beyond E.

New Authority comparisons:

A0:
    N

A1_raw:
    N * E_raw_PGM_rank

A1:
    N * E_cal

A2:
    N * T_v2

PRIMARY:
    A2 vs A1

SECONDARY:
    A2 vs A0
    A2 vs A1_raw

Do not use MVTec to choose this formulation.

======================================================================
Z. AUTHORITY-V2 SAFETY METRICS
======================================================================

Because positive intervention is rare and harmful directions are common,
Authority must NOT be judged only by AUROC.

For each class report:

1. utility-positive AUROC
2. signed utility Spearman
3. mean signed utility among top:
       1%
       2%
       5%
       10%
4. harm fraction among top:
       1%
       2%
       5%
       10%
5. positive-utility yield at same coverages
6. risk-coverage curve
7. A2 vs A1 matched-coverage harm difference
8. A2 vs A1 matched-coverage positive-utility yield difference

Define harm:

    u_signed < -1e-8

Define useful positive:

    u_signed > +1e-8

No production threshold is selected.

No correction is executed.

======================================================================
AA. AUTHORITY-V2 DEVELOPMENT EVIDENCE LADDER
======================================================================

Primary class effect:

    delta_A =
        AUROC(A2) - AUROC(A1)

Catastrophic AUROC warning:

    <= -0.03 per class

Also treat a clear increase in harm at matched coverage as a safety warning.

SUPPORTED by discrimination route:

    mean delta_A >= +0.005
    median >= 0
    >= 8/12 positive classes
    no catastrophic tail

SUPPORTED by selective-safety route:

    mean delta_A >= -0.002
    AND
    at 5% matched coverage:
        mean absolute harm-rate reduction >= 0.02
    AND
        >= 8/12 classes are non-worse in harm
    AND
        positive-utility yield is not materially degraded
    AND
        no catastrophic AUROC/safety tail

PROMISING_BUT_UNCERTAIN:

    either:
        mean delta_A > 0
        median >= 0
        >= 7/12 positive classes

    or:
        delta_A is approximately non-inferior (>= -0.002 mean)
        AND matched-coverage harm reduction >= 0.01 absolute
        AND >= 7/12 classes safety-non-worse

    with no catastrophic tail.

WEAK_POSITIVE_EVIDENCE:

    mean delta_A >= 0
    median >= 0
    OR consistent small safety improvement,
    but magnitude/uncertainty is insufficient for PROMISING.

INCONCLUSIVE:
    near-zero/mixed and no reliable safety gain.

FALSIFIED:
    material degradation or harmful tail.

Again:
    significance alone is not enough.
    perfection is not required.

======================================================================
AB. STABLE-BUT-WRONG V2 AUDIT
======================================================================

Reuse the old conceptual warning but do not blindly copy the old threshold as
the sole Trust test.

Report old-style descriptive rows where:

    E_raw >= 0.75
    selected T_v2 high according to within-image percentile >= 0.75
    patch GT is negative

But additionally stratify by:

    peer_coherence quartile
    query_support_mean quartile
    eigen_entropy quartile
    stage_disagreement quartile
    S9 quartile
    R9 quartile
    S16/R16 where valid

Goal:

    identify whether Trust-v2 quality features actually separate false
    high-evidence cases.

Do NOT use these strata to tune thresholds after inspection.

If writing a large per-patch artifact:
    preflight file size.

If > 80 MB:
    use exact-path Git LFS BEFORE commit.

Never repeat the previous >100 MB normal-blob mistake.

======================================================================
AC. REFERENCE CONTAMINATION
======================================================================

After GT is allowed:

report:
    any anomalous p1-p8 peer fraction
    multiple anomalous peer fraction
    p9 anomaly occupancy
    p16 anomaly occupancy where valid

But explicitly distinguish:

    anomalous-peer contamination

from:

    semantically unsuitable but GT-normal peer cloud.

Do not claim reference contamination explains Trust failure unless data
supports it.

======================================================================
AD. ADVERSARIAL REVIEW QUESTIONS
======================================================================

Before accepting any result, answer explicitly:

1. Did a Trust-v2 feature accidentally contain GT?
2. Did p16 use the same candidate ordering?
3. Was any perturbation independently reranked?
4. Did p16 change K from 8?
5. Did candidate filters change?
6. Did PGM formula change?
7. Did PCRR receive a hidden rescue?
8. Did StandardScaler see held-out class data?
9. Did logistic training see held-out class data?
10. Did MVTec influence model selection?
11. Did medical data get accessed?
12. Is E mandatory in every Trust comparison?
13. Are improvements due only to probability recalibration?
14. Does selected Trust-v2 improve E after class-wise control?
15. Are one/two classes dominating the mean?
16. Is there a negative-tail class?
17. Does safety improve at matched coverage?
18. Does T_v2 merely duplicate E?
19. Does p16 genuinely add information beyond p9?
20. Does PCRR disagreement genuinely add information or only noise?
21. Was any result gate changed after results were seen?
22. Were repeated runs used to cherry-pick randomness?
23. Is the study correctly labeled as VisA development, not independent
    confirmation?

Write:

    ADVERSARIAL_REVIEW.md

======================================================================
AE. VIS-A DEVELOPMENT ARTIFACTS
======================================================================

Produce at minimum:

runs/phase5/sabra/TRUST_V2_DEVELOPMENT/

    SABRA_TRUST_V2_PROTOCOL.md
    SABRA_TRUST_V2_PROTOCOL.json
    READINESS_AUDIT.json
    TRUST_V2_GT_FREE_MANIFEST.json
    P16_COVERAGE_AUDIT.json
    BASELINE_PARITY_AUDIT.json
    P16_GEOMETRY_PARITY_AUDIT.json
    TRUST_V2_MODEL_AUDIT.json
    PCRR_DISAGREEMENT_AUDIT.json
    AUTHORITY_V2_AUDIT.json
    REFERENCE_CREDIBILITY_AUDIT.json
    STABLE_BUT_WRONG_V2.csv
    PER_CLASS_TRUST_V2.csv
    STATISTICS.json
    ADVERSARIAL_REVIEW.md
    DECISION.json
    REPORT.md

If any file exceeds GitHub normal blob limits:
    use exact-path Git LFS before committing.

Do not track broad "*.csv" unless absolutely necessary.
Prefer exact-path .gitattributes entries.

======================================================================
AF. VIS-A DEVELOPMENT DECISION
======================================================================

Produce one of:

    TRUST_V2_DEV_SUPPORTED
    TRUST_V2_DEV_PROMISING
    TRUST_V2_DEV_WEAK_POSITIVE
    TRUST_V2_DEV_INCONCLUSIVE
    TRUST_V2_DEV_FALSIFIED
    TRUST_V2_DEV_INVALID

External MVTec freeze is allowed if:

    implementation integrity PASS

AND

    selected Trust-v2 >= WEAK_POSITIVE_EVIDENCE

AND

    Authority-v2 is not FALSIFIED

AND

    no catastrophic reference/safety failure exists.

This is intentionally permissive enough to preserve weak but coherent positive
evidence.

It does NOT authorize full SABRA training yet.

======================================================================
AG. VIS-A RESULT COMMIT / PUSH
======================================================================

Before commit:
    fetch remote
    remote-race guard
    preflight >80 MB files
    verify LFS pointers if used
    verify no normal Git blob >100 MB

Stage exact paths only.

Commit:

    phase5: audit SABRA Trust-v2 on VisA

Push normally.

Verify:
    LOCAL_HEAD == REMOTE_HEAD
    divergence 0 0

If development is INCONCLUSIVE/FALSIFIED:
    do NOT access MVTec.
    produce final handoff and stop.

If development >= WEAK_POSITIVE_EVIDENCE:
    proceed to candidate freeze.

======================================================================
AH. FREEZE THE FINAL TRUST-V2 CANDIDATE BEFORE MVTEC
======================================================================

The freeze must include:

    exact selected model:
        M1 / M2 / M3
    whether D_rel retained
    exact feature order
    scaler mean/std fit on ALL VisA development classes
    logistic coefficients
    intercept
    random seed
    sklearn version
    formulas
    hashes
    p16 rule
    PGM/PCRR source hashes
    Phase2B hashes
    Need C1 frozen parameters
    Authority formula
    evaluation metrics
    MVTec decision rules

Write:

    runs/phase5/sabra/TRUST_V2_DEVELOPMENT/
        TRUST_V2_FROZEN_MODEL.json
        NEED_C1_FROZEN_MODEL.json
        EXTERNAL_VALIDATION_FREEZE.json

Commit:

    phase5: freeze SABRA Trust-v2 candidate

Push.

Verify remote equality.

ONLY AFTER THIS PUSH may MVTec be read.

======================================================================
AI. EXTERNAL MVTEC VALIDATION — ZERO TUNING
======================================================================

MVTec is external INDUSTRIAL validation.

Absolutely no:
    fitting scaler on MVTec
    fitting Trust coefficients on MVTec
    fitting Need on MVTec
    changing selected features
    changing p16
    changing PGM/PCRR
    changing evidence ladder
    changing Authority formula
    threshold selection
    correction-magnitude selection

Use frozen VisA-trained:
    Trust-v2 model
    Need C1 model

Build MVTec relational quantities using the same within-image GT-free
construction.

MVTec GT may be used only for evaluation.

Medical remains untouched.

======================================================================
AJ. MVTEC EXTERNAL STATUS
======================================================================

Apply the same conceptual evidence ladder using class proportions instead of
hardcoding 12-class counts.

For Trust-v2:

SUPPORTED:
    mean delta >= +0.010
    median >= +0.005
    >= 2/3 classes positive
    no catastrophic tail

PROMISING_BUT_UNCERTAIN:
    mean delta >= +0.005
    median >= 0
    >= 60% classes positive
    no catastrophic tail

WEAK_POSITIVE_EVIDENCE:
    mean > 0
    median >= 0
    >= 50% classes non-negative
    no catastrophic tail

INCONCLUSIVE / FALSIFIED:
    analogous to the frozen VisA rules.

For Authority:
    evaluate discrimination AND matched-coverage safety using the frozen
    rules.

Do NOT require perfect reproduction of VisA effect size.

External validation asks:
    same direction?
    practically useful?
    safe?
    not dominated by a few classes?

======================================================================
AK. FULL-TRAINING AUTHORIZATION AFTER EXTERNAL VALIDATION
======================================================================

This prompt DOES NOT perform full SABRA training.

It only decides whether the next full-training study is justified.

Set:

    NEXT_FULL_SABRA_TRAIN_STUDY_AUTHORIZED=true

only if:

1. implementation/integrity gates PASS;
2. VisA Trust-v2 development >= PROMISING_BUT_UNCERTAIN
   OR VisA is WEAK_POSITIVE_EVIDENCE with strong safety evidence;
3. MVTec Trust-v2 >= PROMISING_BUT_UNCERTAIN;
4. MVTec Authority >= PROMISING_BUT_UNCERTAIN;
5. Need remains usable without external retraining;
6. no catastrophic negative/safety tail;
7. no medical data accessed.

If VisA is merely WEAK_POSITIVE and MVTec is merely WEAK_POSITIVE:
    do NOT authorize full training.
    preserve as research signal only.

======================================================================
AL. EXTERNAL RESULT ARTIFACTS
======================================================================

Under:

runs/phase5/sabra/TRUST_V2_EXTERNAL_MVTEC/

produce:

    READINESS_AUDIT.json
    FREEZE_VERIFICATION.json
    TRUST_V2_EXTERNAL_AUDIT.json
    AUTHORITY_V2_EXTERNAL_AUDIT.json
    PER_CLASS_MVTEC.csv
    STATISTICS.json
    SAFETY_AUDIT.json
    ADVERSARIAL_REVIEW.md
    DECISION.json
    REPORT.md

======================================================================
AM. EXTERNAL RESULT COMMIT / PUSH
======================================================================

Preflight:
    git fetch
    remote-race guard
    file size audit
    LFS audit
    exact-path staging

Commit:

    phase5: validate SABRA Trust-v2 on MVTec

Push normally.

Verify:
    local == remote
    divergence 0 0

======================================================================
AN. MANDATORY HANDOFF
======================================================================

Always create a handoff whether:
    supported
    weak
    inconclusive
    falsified
    invalid
    external data unavailable
    interrupted

Create:

    PROJECT_HANDOFF_SABRA_TRUST_V2_20260819.md
    NEXT_MACHINE_START_SABRA_TRUST_V2.md

and:

runs/phase5/sabra/HANDOFF_TRUST_V2_20260819/

    HANDOFF_STATE.json
    ARTIFACT_MANIFEST.json
    SHA256SUMS.txt
    ENVIRONMENT_SNAPSHOT.txt
    GIT_PROVENANCE.txt
    RESUME_STATUS.md

Record:

    previous SABRA terminal
    current Trust-v2 terminal
    source HEAD
    protocol commit
    implementation commit
    VisA result commit
    freeze commit
    MVTec result commit if any
    branch
    local/remote heads
    divergence
    asset hashes
    cache hashes
    LFS objects
    component statuses
    selected features/model
    PCRR retained/drop
    p16 eligibility
    VisA evidence
    MVTec evidence if run
    full-training authorization
    medical read count
    MVTec-before-freeze count
    next allowed action
    DO_NOT_RERUN instructions

Handoff commit:

    phase5: preserve SABRA Trust-v2 handoff

If interrupted:

    phase5: preserve SABRA Trust-v2 interrupted state

If invalid:

    phase5: preserve SABRA Trust-v2 invalidation handoff

Push normally.

Final verify:

    local HEAD == remote HEAD
    divergence 0 0

======================================================================
AO. POLLING / TOKEN / COMPUTE DISCIPLINE
======================================================================

DO NOT continuously poll long-running jobs.

Forbidden patterns:

    watch ...
    while true ...
    sleep 60; ps ...
    repeated nvidia-smi
    repeated tail every minute
    repeated git status with no event
    repeated identical tests
    repeated result recomputation
    repeated protocol summaries

If a long synchronous command is running:
    let it run.

If detached execution is unavoidable:
    check only at meaningful natural checkpoints.

Do NOT poll more frequently than roughly 10-15 minutes unless the process
itself reports completion/error.

Prefer process completion signals over polling.

Natural checkpoints only:

    START_READY
    PROTOCOL_FROZEN
    IMPLEMENTATION_PARITY_READY
    TRUST_V2_GT_FREE_CACHE_FINALIZED
    IMPLEMENTATION_PUSHED
    VISA_DEV_COMPLETE
    VISA_RESULT_PUSHED
    TRUST_V2_CANDIDATE_FROZEN
    MVTEC_EXTERNAL_STARTED
    MVTEC_EXTERNAL_COMPLETE
    EXTERNAL_RESULT_PUSHED
    HANDOFF_PUSHED

At each checkpoint perform ONE compact self-review:

    provenance
    leakage
    semantics
    parity
    remote state
    rerun necessity

Do not narrate every command.
Do not repeatedly summarize the prompt back to me.

======================================================================
AP. ERROR HANDLING
======================================================================

Stop rather than improvise if:

    source hash mismatch
    checkpoint mismatch
    config mismatch
    CLIP mismatch
    baseline parity mismatch
    p16 compact/direct mismatch
    GT leakage before cache finalization
    MVTec accessed before candidate freeze
    medical data accessed
    remote advanced unexpectedly
    LFS failure
    cache corruption
    result-changing semantic bug after GT
    OOF leakage
    unexpected Phase2B parameter mutation

Use explicit terminal labels such as:

    TRUST_V2_BASELINE_PARITY_FAIL
    TRUST_V2_P16_PARITY_FAIL
    TRUST_V2_GT_LEAKAGE
    TRUST_V2_OOF_LEAKAGE
    MVTEC_ACCESSED_BEFORE_FREEZE
    REMOTE_ADVANCED_DURING_TRUST_V2
    TRUST_V2_STUDY_INVALID

If a result-changing semantic bug is discovered AFTER VisA GT results:
    do not silently patch and rerun.
    invalidate the current study.
    preserve artifacts.
    write a fresh protocol for a future study.

======================================================================
AQ. FINAL OUTPUT TO USER
======================================================================

At completion report compactly:

1. actual starting remote SHA
2. final branch SHA
3. protocol commit
4. implementation commit
5. VisA result commit
6. frozen candidate commit if applicable
7. MVTec external result commit if applicable
8. handoff commit
9. divergence
10. PGM status
11. Need status
12. selected Trust-v2 model
13. p16 coverage/status
14. PCRR disagreement RETAIN/DROP
15. Trust-v2 VisA status and primary effect
16. Authority-v2 VisA status
17. MVTec Trust-v2 status if run
18. MVTec Authority status if run
19. safety summary
20. final literal:

        NEXT_FULL_SABRA_TRAIN_STUDY_AUTHORIZED=true|false

21. medical reads
22. MVTec reads before freeze
23. any blockers

Do not hide negative or weak results.
Do not call WEAK_POSITIVE_EVIDENCE "SUPPORTED".
Do not call development data external validation.
Do not reopen the old SABRA conclusion.

Start now with repository/provenance/remote audit.
Do not ask me to reconfirm normal fast-forward commits or pushes.