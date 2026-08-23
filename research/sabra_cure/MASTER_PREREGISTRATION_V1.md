# SABRA-CURE Master Preregistration v1

Status: `FROZEN_BEFORE_IMPLEMENTATION_AND_RESULTS`

Phase-0 parent: `48cd72b4609200d0a03d9ba3818f61b887c8ab1e`

Phase-0 decision: `PASS_PREREGISTERED`

This protocol is prospective. It does not reinterpret or amend SABRA-CAR history. R0 remains positive oracle evidence, R1-v1 remains a computational stop, and R1-v2 remains a scientific stop. No SABRA-CURE scientific fit or result existed when this document was written.

## 1. Hypothesis and stages

**Hypothesis:** frozen GT-free Phase2B/SABRA evidence can predict continuous signed counterfactual utility and its error under source-class shift; an uncertainty-normalized conservative utility policy can abstain from unsafe signs and apply a bounded correction without target tuning.

- R1: can GT-free features predict signed utility under VisA LOCO?
- R2: can uncertainty select safe actions at useful coverage and improve native source performance?
- R3: can predicted utility magnitude safely parameterize strength beyond sign-only correction?
- R4: can the fully frozen source-derived mechanism improve the one authorized development benchmark?

Each stage is terminal on failure. Later stages require explicit user authorization.

## 2. Immutable evidence and data roles

The parent SHA, R0/R1 artifacts, canonical SABRA source cache, immutable Trust-v2 cache, Phase2B E10 checkpoint, class inventory, manifests, and hashes recorded in `results/sabra_cure/phase0/PROVENANCE.json` are immutable inputs.

VisA is source development only: twelve-class LOCO training/evaluation and source-only selection. MVTec is forbidden through R3 and may be opened once only in an explicitly authorized R4. Medical datasets are forbidden until an R4 pass and a separately committed immutable freeze; they may then be used once only as retrospective benchmarks. No target sample, path, label, mask, statistic, prediction, or metric may affect R1–R3.

Phase2B checkpoint and parameters remain frozen; Phase2B optimizer steps are exactly zero. Feature extraction uses the existing caches and requires zero additional CLIP forwards.

## 3. Inputs and exact feature order

Every patch uses these 14 finite GT-free values, in order:

1. `margin_within_image_rank`
2. `robust_margin_normalization`
3. `D_rank`
4. `deployment_sensitivity`
5. `E`
6. `peer_coherence`
7. `query_support_mean`
8. `peer_eigen_entropy`
9. `stage_query_profile_disagreement`
10. `where(valid_p9,S9,0)`
11. `where(valid_p16,S16,0)`
12. `signed_native_margin = mean_s native_margins[s]`
13. `cross_stage_signed_margin_difference = native_margins[stage2] - native_margins[stage0]`
14. `robust_peer_signed_margin_consensus = median_k mean_s native_margins[s,peer_k]`, falling back to feature 12 only when frozen `valid_b1` is false.

Peer indices and validity come only from the immutable Trust cache. Flattening is class inventory order, cache image order, then ascending patch index. Labels, masks, target data, class name, path text, and patch coordinates are not model inputs.

For every outer fold, feature medians and NumPy-linear Q25/Q75 are fit on the eleven training classes only. IQR is `max(Q75-Q25,1e-6)` elementwise; transform is `(x-median)/IQR`. No clipping, imputation, subsampling, class truncation, or sample weighting is permitted.

## 4. Supervision target

The raw target is committed R0 `u_q=-dL/d(delta_q)`. For each outer fold, compute `s=max(P75(abs(u_train)),1e-8)` on its eleven training classes only, then `y=tanh(u/s)`. Apply that training scale to held-out utility only for evaluation. This preserves sign and bounds targets to `[-1,1]`.

Oracle action is SUPPRESS when `u<-1e-8`, KEEP when `|u|<=1e-8`, and BOOST when `u>1e-8`. It is used for source evaluation only, never as an input.

## 5. Selected model, loss, and numerical policy

H3 is two deterministic linear ridge heads with intercepts:

1. Mean head: `mu=x beta_mu+b_mu`, minimizing sum squared error on `y` plus `1.0*||beta_mu||_2^2`; the intercept is unregularized.
2. Uncertainty head: `z=x beta_z+b_z`, minimizing sum squared error on `log(abs(y-mu_cf)+1e-4)` plus `1.0*||beta_z||_2^2`; the intercept is unregularized. Output `sigma=exp(clip(z,log(1e-4),log(4)))`.

`mu_cf` is strictly cross-fitted inside each outer fold: for each of its eleven training classes, fit the identical mean head on the other ten classes and predict only that excluded class. Fit the uncertainty head to the concatenated eleven inner held-class residual targets. Refit the reported mean head once on all eleven outer-training classes. Inner preprocessing and utility scale are fit on the ten inner-training classes only when producing `mu_cf`; the final uncertainty-head feature preprocessing is the outer-training preprocessing.

Both heads use the centered closed-form normal equations in float64: `(Xc.T@Xc + I)^-1 Xc.T yc`, implemented with `numpy.linalg.solve`, never an explicit inverse. Accumulated sufficient statistics must be mathematically equivalent to the full matrix within `1e-10` on a deterministic fixture. There is no iterative optimizer, stochastic initialization, early stopping, or fallback solver. Seed is 0 for ordering/audits; fitting is deterministic.

Any singular solve, exception, non-finite statistic/parameter/prediction, alignment mismatch, or parity failure is `ENGINEERING_STOP`, not science. A correctness-only fix with a regression test may continue under this protocol only if inputs, formulas, precision, folds, parameters, grids, gates, and outputs are unchanged. Any scientific or numerical-policy change requires a new preregistration.

## 6. R1 protocol and gate

Run exactly twelve outer LOCO folds on all eligible patches. Persist parameters, preprocessing, scale, inner-fold provenance, predictions, residuals, timings, hashes, and per-class metrics.

Primary mean metrics are per-class Pearson correlation, MAE, zero-predictor MAE, and informative sign accuracy. “Informative” uses `|y| >= P50(|y_train|)` from the outer training classes. R1 passes only if:

- median per-class Pearson correlation is at least 0.20;
- at least 9/12 class correlations are strictly positive;
- macro MAE is at least 10% lower than the macro zero-predictor MAE;
- macro informative-sign accuracy is at least 60%, with at least 9/12 classes at or above 50%;
- all provenance/correctness gates pass.

These are utility-prediction gates, not pAP claims. Failure is `R1_SCIENTIFIC_STOP`; do not run R2.

## 7. R2 risk and abstention

For every OOF patch define `r=abs(mu)/sigma`. The H3 policy for `k` acts only when `r>k`: direction is sign(mu); otherwise KEEP. Evaluate exactly `k in {0.5,1.0,1.5,2.0,3.0}`. The unfiltered comparator acts on every nonzero `mu`. Select the numerically lowest `k` satisfying all three source-only risk conditions:

- coverage at least 10%;
- opposite-sign rate among acted patches at most 5%;
- opposite-sign rate at least 25% relatively lower than unfiltered.

No pAP enters threshold selection. Exact ties use higher coverage then lower `k`. If no `k` qualifies, `R2_SCIENTIFIC_STOP`.

Uncertainty must also be informative: predicted sigma has positive residual Spearman correlation in at least 9/12 classes and median correlation at least 0.10; below-median-sigma MAE must be lower than above-median-sigma MAE in at least 9/12 classes. At the selected H3 coverage, the H2 ablation ranks patches by `abs(mu)` with stable patch-index tie breaking. H3 must reduce opposite-sign rate by at least 10% relatively versus H2 while pAP is no worse by more than 0.10 percentage points. Otherwise stop; do not silently fall back to H2.

For the selected policy, apply fixed sign-only correction `delta=direction*0.25*19.840438842773438` on the anomaly channel, zero on the normal channel, identically at all three stages. R2 additionally requires macro pAP at least +0.50 pp over native, macro pAUROC decline no worse than -0.50 pp, and at least 7/12 classes with non-negative pAP delta. Exact canonical pAP/pAUROC implementations from R0 are reused; missing metrics remain null and fail required gates.

## 8. R3 magnitude / radius

Only R2-certified actions are eligible. The sole magnitude proposal is `rho=clip(abs(mu),0,1)`. Compare:

- fixed: `delta=direction*1*0.25*19.840438842773438`;
- utility magnitude: `delta=direction*rho*0.25*19.840438842773438`.

Thus `|delta|<=4.960109710693359`; zero correction reproduces native deployment exactly within the existing audited tolerance. No radius classifier or lambda grid is authorized.

Magnitude passes only if versus fixed it improves macro pAP by at least +0.50 pp, declines macro pAUROC by no more than 0.25 pp, has non-negative pAP delta in at least 8/12 classes, and does not increase opposite-sign risk. If it fails, record a valid negative radius result and retain fixed strength only if the complete R2 gate still holds; otherwise stop.

## 9. R4 development gate

Only after R1–R3 artifacts are committed/pushed and explicit approval is given, freeze the source-derived mechanism and run one MVTec development evaluation. No MVTec-derived tuning is allowed. Versus native Phase2B, R4 requires macro pAP +0.50 pp or more; macro pAUROC decline no worse than -0.50 pp; iAUROC and iAP declines no worse than -1.00 pp each; canonical weighted score non-decline; and non-negative per-class pAP in at least 10/15 classes. Failure is terminal.

After an R4 pass, an immutable freeze must record code/data/checkpoint hashes, feature order, all parameters, thresholds, policy, precision, and dataset roles and must be committed/pushed from a clean tree. Only then, under separate authorization, may the six established Medical retrospective benchmarks run once with no tuning.

## 10. Provenance, leakage, and stop contract

Every stage records Git SHA, artifact SHA256, package versions, class/image/patch order, patch counts, float precision, configuration, parameters, and zero forbidden reads. Training-fold statistics must never use an outer or inner held class. Phase2B training steps and new CLIP forwards remain zero.

MVTec and Medical reads are exactly zero through R3. Any forbidden access is `DATA_FIREWALL_VIOLATION` and terminates the workflow. Any target-informed threshold, architecture, feature, scale, repair, or rerun is forbidden. There is one execution per authorized stage; scientific failures cannot be retried under this protocol. Engineering recovery is limited to semantics-preserving correctness defects with permanent evidence and regression tests.

No expected performance is preregistered. No gate may be relaxed after results.
