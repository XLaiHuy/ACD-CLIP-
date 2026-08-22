# SABRA-CAR Master Preregistration

Status: FROZEN BEFORE CAR EXPERIMENTS
Preregistered at: 2026-08-23T01:01:24+07:00
Repository: XLaiHuy/ACD-CLIP-
Baseline branch: `research/p5-sabra-canonical-v1`
BASE_SHA: `a986bcfee41c31f03d38e449efb8826d56c90525`
CAR branch: `research/p6-sabra-car-v1`
Scientific implementation ancestor: `4aa9b465ddeb072e9218b74982306d6324c62375`

## Scientific objective and invariant

SABRA-CAR tests Counterfactual Action Radius in the fixed order Direction,
Trust/Stability, Radius, Integration, Freeze, and final evaluation. The primary
metric is pixel average precision (pAP). The published ACD-CLIP N=3 reference is
pAUROC 91.55, pAP 43.03, iAUROC 77.90, and iAP 77.70 percent.

No threshold, split, seed, metric, class inventory, feature order, candidate
strength, or model family below may be changed after observing its stage result.
Engineering defects may be fixed with a documented regression test and separate
commit; a scientific failure may only follow a fallback written here.

## Frozen identities and data roles

- Phase2B is frozen E10 at
  `/home/ai4/caohuy/acdclip_runs/canonical_sabra_v1_seed0/phase2b/checkpoints/adapter_10.pth`,
  SHA256 `6643cd68eafabf9acdb724242ef5b2d1fbc4bf7e9d2ba7ad6c47776ea646da80`.
- The canonical exact deployment is shared abnormal-logit correction over all
  three stages, zero normal-channel delta, followed by canonical resize, blur,
  stage mean, and two-class softmax. The classification branch is unchanged.
- VisA's fixed 2,162-row metadata and 12-class inventory is source scientific
  data. Its GT may be opened only by oracle/target/evaluation code after the
  GT-free evidence cache is finalized.
- MVTec AD is development-only. It may be used once per explicitly allowed
  selection stage and never after freeze.
- The six Medical datasets are retrospective final benchmarks because prior
  SABRA-v1 results informed the hypothesis. They may not be opened before a
  successful freeze and may never drive a post-freeze change.
- No untouched external benchmark is currently claimed. Unless a protocol-legal
  one is already locally identified before freeze,
  `UNTOUCHED_EXTERNAL_VALIDATION=PENDING`.
- Existing caches may be reused only after checkpoint, CLIP, config, metadata,
  record count, class inventory, field contract, and cache-shard hashes match.

All percentages below are percentage points (pp). Macro metrics are unweighted
means over the 12 VisA classes or the applicable development/final datasets.
Ties fail any strict improvement gate. Undefined image metrics remain null and
are excluded only from the macro for that metric; they are never converted to
zero.

## Common correctness gates

Every stage requires: deterministic unit tests PASS; finite values; exact class
inventory; no Phase2B parameter gradients or updates; zero Phase2B training
steps; exact checkpoint/config/CLIP/metadata provenance; normal-channel delta
identically zero; shared correction identical across stages; canonical metric
and deployment parity PASS; and Medical reads equal zero before freeze. A failed
correctness gate is an engineering stop/fix, not a scientific result.

## R0 — Signed Direction

For each VisA image and patch, compute the exact canonical Focal+Dice gradient
at zero shared abnormal-logit intervention:

`u = - d L_Focal+Dice / d delta`, with epsilon `1e-8`.

Labels are BOOST for `u > 1e-8`, KEEP for `abs(u) <= 1e-8`, and SUPPRESS for
`u < -1e-8`. Evaluate the fixed alpha landscape `{0, 0.125, 0.25, 0.5, 1.0}`,
where the logit correction magnitude is `alpha * s_m`, and
`s_m=P90(abs(native_margin))` from VisA (`19.840438842773438` for the verified
canonical cache). The four preregistered variants are:

1. Native Phase2B: delta zero.
2. Positive-only oracle: BOOST patches receive `+alpha*s_m`; all others KEEP.
3. Signed oracle: BOOST receives `+alpha*s_m`, SUPPRESS receives
   `-alpha*s_m`, KEEP receives zero, with one global alpha chosen by macro pAP.
4. Signed-radius oracle: each non-KEEP patch independently chooses the best
   action among `{0, 0.125, 0.25, 0.5, 1.0}*sign(u)*s_m` by lowest per-image
   canonical loss; ties choose the smaller magnitude. This is oracle analysis,
   not a deployable policy.

Positive-only and signed comparisons use the same selected alpha. Select alpha
by signed-oracle macro pAP; ties use lower alpha. Report native and every alpha
without suppression. Report macro/per-class pAP and pAUROC, Focal+Dice loss,
BOOST/KEEP/SUPPRESS rates, positive-to-suppress sign reversal rate, and counts
for all four loss/AP direction quadrants.

R0 passes only if all common gates pass, signed oracle improves macro pAP over
matched positive-only by at least +1.00 pp, at least 8/12 classes improve, and
macro pAUROC is no worse by more than 0.50 pp. If signed intervention has at
least +1.00 pp pAP headroom over native with at least 8/12 classes and safe
pAUROC, but fails G1 while at least 50% of per-image non-tied interventions
show loss/AP disagreement, trigger R0B. Otherwise stop at R0.

## R0B — Ranking-aware fallback

R0B is the only direction fallback. Before its implementation commit, append
and commit the exact mathematical definition. The frozen allowed family is a
deterministic pairwise logistic ranking loss on source GT only. For each image,
take anomaly pixels and normal pixels in the top decile of native abnormal
score; deterministically form equal-count pairs by stable score order and use
`L_rank = mean(softplus(score_normal - score_anomaly))`. If either set is empty,
the image contributes no pairs. Define `u_rank=-dL_rank/ddelta` at delta zero,
using the same shared abnormal-logit intervention and epsilon.

Evaluate exactly the R0 alpha grid and diagnostics. R0B passes if ranking-signed
beats the canonical-loss signed direction by at least +0.50 pp macro pAP,
beats native by at least +1.00 pp, improves at least 8/12 classes, and loses no
more than 0.50 pp macro pAUROC. Otherwise stop. If it passes, `u_rank` is the
sole target used downstream.

## R1 — GT-free action predictor

Target is the certified three-class source-oracle action. Model family is only
multinomial logistic regression with `C=1`, `solver=lbfgs`, `class_weight=balanced`,
`max_iter=1000`, `random_state=0`; features are standardized using training-fold
median and IQR (IQR floor `1e-6`). The frozen feature order is:

1. `margin_within_image_rank`
2. `robust_margin_normalization`
3. `D_rank`
4. `deployment_sensitivity`
5. `E`
6. `peer_coherence`
7. `query_support_mean`
8. `peer_eigen_entropy`
9. `stage_query_profile_disagreement`
10. supported p9 stability indicator
11. supported p16 stability indicator

Use 12-fold leave-one-class-out VisA predictions only. Confidence is maximum
class probability. The selective predictor acts when confidence is at least a
threshold from `{0.50,0.60,0.70,0.80,0.90}` and otherwise predicts KEEP. Select
the lowest threshold meeting the risk gate; ties use higher coverage.

R1 passes when: action coverage is at least 10%; opposite-sign error among
acted patches is at most 5%; opposite-sign error is at least 25% relatively
lower than the unfiltered non-KEEP argmax predictor; LOCO CAR intervention
improves macro pAP over native by at least +0.50 pp; macro pAUROC decline is at
most 0.50 pp; and at least 7/12 classes have non-negative pAP delta. If no fixed
threshold passes, stop. No neural predictor fallback is authorized.

## R2 — Trust and directional Stability

At the R1-selected action coverage, compare exactly: Direction; Direction +
Trust; Direction + Stability; Direction + Trust + Stability. Trust uses the
existing fixed features in this order: `E`, `peer_coherence`,
`query_support_mean`, `peer_eigen_entropy`, and
`stage_query_profile_disagreement`. Stability consists only of the supported
p9/p16 diagnostics already present in the cache. Any logistic reliability fit
uses the same fixed settings and LOCO discipline as R1. Coverage matching uses
stable sorting by score then patch index and must agree within 0.10 pp absolute.

A reliability component is useful only if, at matched coverage, it reduces
opposite-sign rate by at least 20% relative and harmful-action rate by at least
10% relative versus Direction, while pAP is no worse than Direction by more
than 0.10 pp. Trust must pass or the tree stops. Stability is retained only if
adding it to Direction+Trust further reduces opposite-sign rate by at least
10% relative or harmful-action rate by at least 5% relative, with pAP no worse
by more than 0.10 pp. Otherwise Stability is dropped and Direction+Trust
continues as a valid negative component result.

## R3 — Radius / Strength

Only certified R2 actions are eligible. Compare fixed signed correction against
the discrete radius policy `rho in {0,0.5,1.0}` in
`delta=direction*rho*lambda*source_margin_scale`. Fit a multinomial logistic
radius predictor with the same preprocessing/settings/features/LOCO discipline
as R1. Oracle radius targets choose the lowest-loss rho; ties choose smaller rho.
The global lambda candidates are exactly `{0.125,0.25,0.5,1.0}` and are chosen
on VisA LOCO macro pAP, ties choosing the smaller lambda.

Discrete radius passes if it beats the matched fixed signed correction by at
least +0.50 pp macro pAP, macro pAUROC decline versus fixed is at most 0.25 pp,
at least 8/12 classes are non-negative versus fixed, and opposite-sign risk does
not increase. If it fails, retain fixed signed correction and continue only if
fixed still satisfies R1/R2 safety and gives at least +0.50 pp pAP over native.
Continuous strength is authorized only if the discrete oracle exceeds the
predicted discrete policy by at least +1.00 pp macro pAP. If triggered, use
ridge regression with alpha `1.0`, clip rho to `[0,1]`, and retain it only if it
beats discrete by +0.50 pp with the same safety/breadth gates. No other strength
model or grid is authorized.

## R4 — Integrated development model

Integrate only certified Direction, Trust, optional Stability, and the retained
radius policy. Fit all source predictors using VisA only, then perform one
MVTec development evaluation. The only MVTec-selectable global quantity is
lambda from `{0.125,0.25,0.5,1.0}` if not already fixed by R3; ties use smaller
lambda. No repeated MVTec retuning is allowed.

R4 passes if versus native Phase2B on MVTec: macro pAP improves at least +0.50
pp; macro pAUROC decline is at most 0.50 pp; iAUROC and iAP each decline at most
1.00 pp; the canonical weighted score does not decline; and per-class pAP is
non-negative for at least 10/15 MVTec classes. Otherwise stop at R4.

## Freeze and final evaluation

After R4 passes, write an immutable freeze containing BASE_SHA, Phase2B path
and SHA256, CAR code SHA, exact feature orders and fitted parameters, retained
components, radius policy, lambda, margin scale, thresholds, precision, dataset
roles, and selection protocol. Freeze requires a clean tree, all tests and
provenance checks passing, every scientific file committed, and local HEAD equal
to remote HEAD. Commit and push the freeze. No scientific parameter changes are
permitted afterward.

Only then run the six existing Medical datasets once as
`RETROSPECTIVE_MEDICAL_BENCHMARK`, using the same frozen E10 and CAR artifact.
Report exact per-dataset and macro pAUROC, pAP, iAUROC, iAP; improvement breadth;
worst regression; best improvement; and comparisons to Phase2B, SABRA-v1, and
published ACD-CLIP. `BEATS_ACD_*` uses strict greater-than at full precision.

## Stop, bugs, and reproducibility

A failed scientific gate produces committed `FINAL_DECISION.md` and
`AUTOPILOT_HANDOFF.md`; missing final CSVs are represented by schema-valid files
with explicit `NOT_RUN` status, never fabricated metrics. Each engineering bug
is permanently recorded, receives a regression test and separate fix commit,
and resumes only from provenance-valid artifacts. Every stage records runtime
estimate and finish time before a long command, avoids frequent polling, stages
explicit paths only, runs `git diff --cached --check`, commits, and pushes.
