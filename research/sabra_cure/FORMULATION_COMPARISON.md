# SABRA-CURE Formulation Comparison

## H0–H4 decision

| Candidate | Evidence fit | KEEP / signed-risk mechanism | Complexity and defensibility | Decision |
|---|---|---|---|---|
| H0: 3-class + calibration | Discards continuous R0 magnitude; existing confidence is anti-safety globally and class-dependent | Thresholded max probability | Cheapest, but calibration alone cannot repair a ranking that worsens risk | Comparator only |
| H1: ACT/ABSTAIN then SIGN | Represents KEEP and asymmetric error | Two classifiers and two calibrations | Adds hierarchy and imbalance while discretizing continuous utility | Reject |
| H2: signed utility regression | Direct match to R0 evidence | Sign and magnitude threshold | Simple and interpretable, but no sample-conditional error estimate | Retain as ablation |
| H3: signed utility + uncertainty | Direct match; probe finds residual ordering in 11/12 folds | Conservative utility margin separates value from abstention | Two closed-form linear heads; small, reproducible, auditable | **Select** |
| H4: two value surfaces | Could encode action-specific value | Compare boost/suppress values | R0 provides a directional derivative, not identified separate potential-outcome surfaces; would overclaim supervision | Reject |

H3 is the minimum expansion supported by Phase-0 evidence. The selected model is a two-stage ridge location/uncertainty estimator: ridge prediction of transformed signed utility, followed by ridge prediction of log absolute cross-fitted residual. It has 30 fitted scalar parameters per outer fold (two 14-weight vectors and two intercepts), no iterative optimizer, and no Phase2B or CLIP changes.

## Related-work positioning

| Area | Representative work | What it establishes | SABRA-CURE distinction / hypothesis |
|---|---|---|---|
| Training-free ZSAD | [WinCLIP, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.html) | Window/patch CLIP aggregation and prompt ensembles enable zero-shot anomaly localization | Does not learn a post-hoc signed intervention policy from counterfactual utility |
| Object-agnostic prompting | [AnomalyCLIP, ICLR 2024](https://arxiv.org/abs/2310.18961), [PromptAD, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Li_PromptAD_Zero-Shot_Anomaly_Detection_Using_Text_Prompts_WACV_2024_paper.html) | Auxiliary-source prompt learning improves category/domain transfer | SABRA-CURE freezes Phase2B and changes neither prompts nor encoders |
| Adaptive prompts | [AdaCLIP, ECCV 2024](https://arxiv.org/abs/2407.15795), [AA-CLIP, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Ma_AA-CLIP_Enhancing_Zero-Shot_Anomaly_Detection_via_Anomaly-Aware_CLIP_CVPR_2025_paper.html) | Static/dynamic or anomaly-aware adaptation can improve ZSAD | Proposed novelty is cache-only counterfactual utility prediction with selective signed correction, not another prompt adapter |
| Selective prediction | [SelectiveNet, ICML 2019](https://proceedings.mlr.press/v97/geifman19a.html), [post-hoc confidence study, UAI 2024](https://proceedings.mlr.press/v244/cattelan24a.html) | Abstention should be evaluated by risk–coverage; softmax confidence can be broken | R1-v2 directly exhibits broken confidence; H3 learns uncertainty tied to utility error |
| Selective regression | [Shah et al., ICML 2022](https://proceedings.mlr.press/v162/shah22a.html), [Noskov et al., ACML 2024](https://proceedings.mlr.press/v222/noskov24a.html) | Conditional mean/variance or variance testing can support abstention | Source-only cross-class utility residuals define the uncertainty target; no distribution-free guarantee is claimed |
| Risk control | [Automatically Adaptive Conformal Risk Control, AISTATS 2025](https://proceedings.mlr.press/v258/blot25a.html) | Risk control can adapt to sample difficulty | Inspires separation of risk from utility, but SABRA-CURE uses frozen source LOCO gates, not a conformal guarantee |
| Counterfactual policy learning | [Trustworthy Policy Learning under Counterfactual No-Harm, ICML 2023](https://proceedings.mlr.press/v202/li23ay/li23ay.pdf) | Policies can explicitly constrain counterfactual harm | SABRA-CURE’s “counterfactual” is an audited local deployment intervention, not an observational causal-treatment claim |

We did not identify a directly equivalent formulation in the reviewed literature. This is a bounded positioning claim, not a claim that no prior work exists. The proposed novelty axis is: frozen-ZSAD, cache-only learning of signed local counterfactual utility plus utility-error uncertainty, with source-LOCO selective risk gates and a bounded reversible intervention.

## Research assessment (qualitative, not benchmark results)

Novelty 8/10; adaptability 8/10; performance potential 7/10; scientific coherence 9/10; inference efficiency 9/10; reviewer defensibility 8/10; implementation risk 3/10; overall 8/10.

The strongest unresolved risks are correlated patch pseudo-replication, class-shifted uncertainty calibration, utility saturation, spatially clustered failures, and the possibility that source-side utility predictability does not translate into pAP improvement. The staged gates are designed to stop on each of these rather than tune around them.
