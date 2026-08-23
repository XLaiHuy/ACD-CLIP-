# SABRA-CURE Phase-0 Diagnostic

Status: `DIAGNOSTICS_COMPLETE`

Parent: `48cd72b4609200d0a03d9ba3818f61b887c8ab1e`

Scope: existing VisA source, R0, and R1-v2 artifacts only.

## Evidence contract

- **OBSERVED:** R0 is positive oracle evidence; R1-v1 is a computational stop; R1-v2 converged in all folds and is a scientific stop.
- **DERIVED:** all statistics here are deterministic calculations from committed/canonical source-side artifacts. Exact paths and SHA256 values are in `results/sabra_cure/phase0/PROVENANCE.json`.
- **HYPOTHESIS:** a continuous utility model with separately estimated uncertainty can turn the available signed evidence into a safer policy.
- **SIMULATED:** the 2,048-patch-per-class feasibility probe is only a pathology check and cannot satisfy a scientific gate.
- **LITERATURE_SUPPORTED:** selective regression and risk-coverage methods motivate separating value estimation from abstention; they do not establish SABRA-CURE performance.

## D0: why R1-v2 stopped

Published-result parity passed exactly: predicted counts are SUPPRESS 1,192,010, KEEP 1,451,100, BOOST 316,668. The unfiltered policy covers 50.973% with 11.370% opposite-sign risk. Every frozen confidence cutoff fails: at 0.50, coverage is 24.766% but risk rises to 14.312%; higher cutoffs reduce coverage below 10% and raise risk to 30.342–45.251%.

Confidence is therefore `CLASS_DEPENDENT_NONMONOTONIC`, not a safety score. Among acted patches, its correlation with opposite-sign error is +0.24590 globally, positive in 9/12 classes and negative in 3/12. Opposite-sign errors have higher median confidence (0.5523) than correct signed actions (0.5095). High disagreement, Trust, and stability quartiles have much higher risk than low quartiles. Spatial error adjacency is enriched above independent expectation in every class (3.31x–24.10x). The primary failure is formulation mismatch: maximum class probability ranks decisiveness of a three-class classifier, but not signed intervention safety.

Class/action imbalance is material but not sufficient as a sole explanation: the oracle distribution is SUPPRESS 1,459,383, KEEP 1,379,633, BOOST 120,762; the frozen classifier used balanced class weights yet still produced the unsafe confidence ordering.

## D1: continuous utility

The R0 utility `u_q=-dL/d(delta_q)` is finite, path/order aligned across all 12 classes and 2,959,778 patches, and supported by the committed finite-difference parity audit. No target-domain data are involved.

Three odd, sign-preserving targets were compared without fitting a scientific model. T2, `tanh(u/s)`, with `s=P75(|u|)` from the eleven training classes only and floor `1e-8`, is selected. It preserves sign exactly, is finite, has P90 cross-fold CV 0.0377, median P99/P90 1.061, and maximum saturation fraction 0.1803. T1 and T3 retain much heavier fold-variable tails (P90 CV 0.5085/0.5292; median P99/P90 2.84/5.48).

## D2: features

All 11 frozen R1 features are finite. `deployment_sensitivity` is magnitude evidence: the implementation uses `gradient.detach().abs()`, its IQR is `1.18e-12`, and it hits the `1e-6` scaler floor in all 12 folds. It is retained as existing magnitude evidence, not misrepresented as direction.

Five deterministic cache-only signed candidates were audited. Three survive the prespecified association/redundancy rules: `signed_native_margin`, `cross_stage_signed_margin_difference`, and `robust_peer_signed_margin_consensus`. Query-minus-peer margin and signed relational residual are rejected as redundant with signed native margin (absolute correlation at least 0.95 under the audit rule). No new CLIP forward is required.

## Bounded feasibility probe

The deterministic probe used exactly 2,048 evenly spaced patches per class, seed 0, identical subsets, and closed-form ridge models. It is labeled `PROBE_ONLY_NOT_SCIENTIFIC_RESULT`. Mean-target correlation is positive in 12/12 folds (median 0.4348). Predicted uncertainty orders lower versus higher MAE in 11/12 folds; median uncertainty/residual correlation is 0.2363. This only rejects obvious unlearnability and supports testing H3; it is not evidence of pAP improvement.

## Firewall and execution

No full R1 run, R2, R3, R4, Phase2B training, or CLIP inference occurred. MVTec and Medical access counts are zero.
