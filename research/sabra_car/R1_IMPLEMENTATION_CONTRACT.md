# R1 GT-Free Action Predictor Implementation Contract

Status: FROZEN BEFORE IMPLEMENTATION AND RESULTS

Timestamp: 2026-08-23T02:24:00+07:00

This addendum resolves operational details left implicit by the master
preregistration. It does not change the model family, feature order, folds,
threshold grid, metrics, gates, or stop logic.

## Inputs and provenance

- R0 source-oracle labels are the exact `BOOST=1`, `KEEP=0`, and
  `SUPPRESS=-1` arrays obtained by applying epsilon `1e-8` to the committed
  R0 utility shards.
- Features 1-9 come only from the immutable canonical cache at
  `/home/ai4/caohuy/acdclip_runs/canonical_sabra_v1_seed0/sabra_source/gt_free_cache`.
- Features 10-11 come only from the immutable Trust-v2 GT-free cache at
  `runs/phase5/sabra/TRUST_V2_DEVELOPMENT/cache`.
- Both manifests, every shard hash, the 2,162-record count, the exact 12-class
  inventory, and image-path alignment between caches and R0 utility shards must
  pass before fitting.
- The Trust-v2 manifest must state finalized/immutable, baseline parity PASS,
  p16 parity PASS, GT-free only, and zero Medical/MVTec reads.
- No mask or other GT is opened during feature construction or model fitting.
  VisA masks are opened only by the post-prediction source evaluation.
- Medical reads remain zero. Phase2B training steps remain zero.

## Exact feature matrix

The per-patch feature order is frozen as:

1. `margin_within_image_rank`
2. `robust_margin_normalization`
3. `D_rank`
4. `deployment_sensitivity`
5. `E`
6. `peer_coherence`
7. `query_support_mean`
8. `peer_eigen_entropy`
9. `stage_query_profile_disagreement`
10. `where(valid_p9, S9, 0)`
11. `where(valid_p16, S16, 0)`

The last two are the preregistered supported p9/p16 stability indicators:
unsupported patches are exactly zero rather than imputed. The verified VisA
cache has 100% p9 and p16 validity, but the mask rule remains part of the
frozen deployment contract. All inputs must be finite. Flattening order is
class inventory order, then cache image order, then ascending patch index.

## LOCO preprocessing and estimator

Use twelve leave-one-class-out folds. For each held-out class:

- fit preprocessing on patches from the other eleven classes only;
- compute each training-feature median and the 25th/75th percentiles using
  NumPy's linear quantile definition;
- define `IQR=q75-q25` and replace values below `1e-6` by `1e-6`;
- transform training and held-out features as `(x-median)/IQR`;
- fit one scikit-learn `LogisticRegression` with
  `C=1.0`, L2 penalty, `solver="lbfgs"`, `class_weight="balanced"`,
  `max_iter=1000`, `random_state=0`, `tol=1e-4`,
  `fit_intercept=True`, and `multi_class="multinomial"`;
- train on all eligible training patches; no subsampling or class truncation;
- require estimator classes exactly `[-1,0,1]`, finite parameters and
  probabilities, probability rows summing to one within `1e-6`, and
  convergence before 1,000 iterations.

The already-installed `Thai` environment is frozen for fitting/prediction:
Python 3.14.0, NumPy 1.26.4, scikit-learn 1.7.2. It does not execute canonical
deployment. It writes per-fold medians, IQRs, coefficients, intercepts,
iteration counts, class order, held-out probabilities, predicted class,
confidence, identities, and package versions.

Canonical correction deployment and exact pAP/pAUROC evaluation run only in
the existing `torchhuy` environment used by R0. The two environments exchange
only committed numerical artifacts; no fitted object is unpickled across
environments.

## Prediction, threshold selection, and risk

For each OOF patch, the unfiltered prediction is stable argmax over class
probabilities in estimator class order `[-1,0,1]`; confidence is that maximum
probability.

For threshold `t in {0.50,0.60,0.70,0.80,0.90}`:

- output the argmax action only when confidence is at least `t`;
- otherwise output KEEP;
- predicted KEEP never counts as an acted patch;
- action coverage is predicted non-KEEP patches divided by all patches;
- an opposite-sign error is predicted BOOST with oracle SUPPRESS or predicted
  SUPPRESS with oracle BOOST;
- opposite-sign rate is opposite-sign errors divided by acted patches;
- the unfiltered comparator uses every non-KEEP argmax prediction without a
  confidence threshold and the same denominator definition;
- relative risk reduction is
  `1 - selective_opposite_rate/unfiltered_opposite_rate`; if the unfiltered
  rate is zero, the risk condition passes only when the selective rate is also
  zero and the reported reduction is null.

The risk-qualified set contains thresholds with coverage at least 10%,
opposite-sign rate at most 5%, and relative reduction at least 25% (or the
zero-baseline exception above). Select the numerically lowest qualifying
threshold. Exact duplicate-threshold outcomes retain the one with higher
coverage, then lower threshold. If none qualifies, R1 is a scientific STOP.

## OOF intervention and scientific gates

Use the R0-selected fixed strength only:
`delta = predicted_action * 0.25 * 19.840438842773438`.
The normal channel remains zero and the correction is shared identically across
all three stages. There is no radius prediction in R1.

For the selected threshold, concatenate the twelve held-out predictions and
compute exact per-class/global-pixel pAP and pAUROC with the R0 metric
implementation. Macro values are unweighted class means. A class is
non-negative when its pAP delta versus native is at least zero; exact ties
count as non-negative for the stated R1 breadth gate.

R1 passes only when all correctness/provenance gates pass and:

- coverage is at least 10%;
- opposite-sign rate is at most 5%;
- opposite-sign rate is reduced at least 25% relatively versus unfiltered;
- LOCO CAR macro pAP improves over native by at least +0.50 pp;
- macro pAUROC decline versus native is no worse than -0.50 pp;
- at least 7/12 classes have non-negative pAP delta.

All threshold rows, fold parameters, OOF predictions, class metrics,
preprocessing statistics, convergence records, provenance hashes, and gate
values are written before the decision. Missing results remain missing; no
interpolation, refit, neural fallback, or threshold change is allowed.
