# ACD-CLIP Phase4 P1-v8: Current Architecture & Evidence Report

## Executive Summary & Final Decision

**FINAL DECISION**: `FIX_LOCAL_OBJECTIVE`

- **Vector Specialization (Tier-1)**: `PASS` ($\Delta T$ median cosine $\approx 0.8930$, max $\approx 0.9821$, effective rank $\approx 2.64$)
- **Dense Router Implementation**: `PASS` (Dynamic factor routing and local text projection functional)
- **Tier-2 Loss Wiring & Audit**: `PASS` (Checkpoint variation verified, non-zero gradient flow confirmed to factor heads and dynamic generators)
- **Tier-2 Functional Specialization Method**: `FAIL` (Factor patch-logit correlation remains $\approx 0.999269$; patch logits collapse functionally across factors)
- **Unsupervised Cluster Semantic Usefulness (Tier-3)**: `FAIL` (Plain $k$-means clusters partition mostly normal/background patches; anomaly-share gap between clusters is only $0.677$ percentage points)
- **Responsibility Loss Calibration (Tier-3)**: `FAIL` ($\lambda_{\text{cluster\_resp}} = 0.05$ dominated total loss at $44.00\% - 52.04\%$ share)
- **8-Epoch / 20-Epoch Gate**: `BLOCKED` until mask-supervised strata objective is implemented.

---

## A. Current Architecture Overview

```
                          Input Image / Text Prompts
                                     |
                                     v
                       +---------------------------+
                       |  Hard-Adapted Text Anchor |
                       +---------------------------+
                                     |
                +--------------------+--------------------+
                |                                         |
                v                                         v
   +--------------------------+             +--------------------------+
   |   Global Classification  |             |  Patch & Prototype Feats |
   |  & Main Phase2B DFG Seg  |             +--------------------------+
   +--------------------------+                         |
                |                                       v
                |                           +--------------------------+
                |                           |     CLS Semantic VAE     |
                |                           +--------------------------+
                |                                       |
                |                                       v
                |                           +--------------------------+
                |                           | M=4 Dynamic Text Factors |
                |                           | (hard + delta-T factors) |
                |                           +--------------------------+
                |                                       |
                |                                       v
                |                           +--------------------------+
                |                           |    Dense Patch Router    |
                |                           +--------------------------+
                |                                       |
                |                                       v
                |                           +--------------------------+
                |                           | Patch-Local Anomaly      |
                |                           | Logits (factor-specific) |
                |                           +--------------------------+
                |                                       |
                |                                       v
                |                           +--------------------------+
                |                           |  Bounded Rho Residual    |
                |                           +--------------------------+
                |                                       |
                +--------------------+------------------+
                                     |
                                     v
                       +---------------------------+
                       | Final Segmentation Logits |
                       +---------------------------+
```

### Module Configuration:
1. **Global Branch**:
   - Hard-adapted text anchor ($T_{\text{hard}}$) for global classification.
   - Main Phase2B Dynamic Feature Grouping (DFG) segmentation path.
2. **Local Branch**:
   - Multi-level visual patch features fused with normal/abnormal prototypes.
   - CLS semantic VAE producing latent slot conditioning.
   - $M=4$ image-conditioned dynamic text factors ($T_{\text{factor\_m}} = \text{normalize}(T_{\text{hard}} + \Delta T_m)$).
   - Factor identity embeddings and factor-specific output heads.
   - Dense per-patch router generating continuous routing weights across $M=4$ factors.
   - Bounded $\rho$ residual added directly to main segmentation logits.
3. **Experts & Routing Constraints**:
   - FOFS paired experts: `OFF` (`--no-h6_expert_enabled`)
   - Visual experts: `OFF`
   - Prediction Top-K: `OFF` (`h6_prediction_routing = dense`)

---

## B. Tier 1: Vector Specialization Audit

### Implementation:
- Factor identity embeddings ($E_{\text{factor}} \in \mathbb{R}^{M \times D}$) providing distinct base identity vectors per factor.
- Factor-specific output heads for local text projection.
- $\Delta T$ diversity loss ($\mathcal{L}_{\text{orth}}$) penalizing pairwise cosine similarity between dynamic residuals $\Delta T_m$.
- Hard-anchor / local-factor path separation ensuring global prompts remain stable while local factors adapt.
- Gradient and checkpoint plumbing for all H6 submodules (`h6_state_dict`).

### Evidence & Metrics:
- **$\Delta T$ Cosine Median**: `0.8930` ($< 0.900$ target threshold)
- **$\Delta T$ Cosine Max**: `0.9821`
- **Effective Rank**: `2.64` ($> 2.0$ target threshold)
- **Vector-Specialization Gate**: `PASS`

---

## C. Tier 2: Functional Specialization Audit

### Implementation:
- Functional decorrelation loss ($\mathcal{L}_{\text{func\_div}}$) computed directly on actual per-factor patch logits ($S_{b,p,m}$).
- Ground-truth anomaly + hard-anchor confidence weighting.
- Tested $\lambda_{\text{func\_div}}$ candidates: `1e-4`, `3e-4`, `1e-3`.

### Audit Findings & Evidence:
1. **Checkpoint Variation**: Distinct SHA-256 hashes verified for factor patch logits across candidate runs:
   - Candidate `T2-A` ($\lambda = 1e-4$): `d9bfac1d50923606...`
   - Candidate `T2-B` ($\lambda = 3e-4$): `7028aa2796b56f8a...`
   - Candidate `T2-C` ($\lambda = 1e-3$): `a7a9e6d4bd6c161d...`
2. **Metadata & CLI Plumbing**: Correct candidate checkpoints loaded, CLI arguments validated.
3. **Loss Non-Zero & Scaling**: $\mathcal{L}_{\text{func\_div}} \approx 1.0$, weighted contribution scales directly with $\lambda$.
4. **Gradient Flow**: Verified non-zero gradients reaching factor-specific heads ($5.0 \times 10^{-6}$), factor identity embeddings ($5.9 \times 10^{-7}$), dynamic prompt generator ($6.5 \times 10^{-6}$), and encoded factor text ($4.3 \times 10^{-4}$). No `detach` or `no_grad` bugs present.
5. **Functional Specialization Failure**: Average off-diagonal correlation of patch logits across factors remains $\approx 0.999269$ ($\approx 1.0$).
6. **Verdict**:
   - Tier-2 Implementation: `PASS`
   - Tier-2 Functional-Specialization Method: `FAIL` (Vector diversity alone does not induce distinct spatial feature responses without targeted spatial supervision).

---

## D. Tier 3: Cluster-Responsibility Audit

### Implemented Architecture:
- Bounded patch-feature bank accumulated during training.
- Deterministic $M=4$ $k$-means clustering producing 4 spatial cluster centroids.
- Centroid-conditioned factor IDs, semantic slots, and router keys.
- Soft cluster targets ($q_{\text{cluster}}$) computed per patch.
- Auxiliary responsibility loss: $\mathcal{L}_{\text{resp}} = \text{KL}(q_{\text{cluster}} \parallel \text{dense\_router\_probs})$.

### Coefficient Audit ($\lambda_{\text{cluster\_resp}} = 0.05$):
- **Weighted Loss Shares**:
  - Epoch 1: `51.20%` of total loss ($\mathcal{L}_{\text{resp\_weighted}} = 1.330$ / $\mathcal{L}_{\text{total}} = 2.597$)
  - Epoch 2: `52.04%` of total loss ($\mathcal{L}_{\text{resp\_weighted}} = 1.118$ / $\mathcal{L}_{\text{total}} = 2.148$)
  - Epoch 3: `44.00%` of total loss ($\mathcal{L}_{\text{resp\_weighted}} = 1.011$ / $\mathcal{L}_{\text{total}} = 2.298$)
- **Root Cause**: $\lambda = 0.05$ did **not** represent a 5% loss contribution. Unnormalized KL divergence magnitude ($\mathcal{L}_{\text{raw}} \approx 20 - 26$) caused the auxiliary responsibility objective to dominate main task training.

### Gradient Evidence:
- Router: `0.10410`
- Image Adapter: `0.07526`
- Factor-ID / Semantic-Slot Path: `0.00890`
- Text-Adapter / Projector: `0.00000` (Expected behavior: KL loss acts directly on router and slot keys, not on unrelated text projections; not a wiring bug).

### Cluster Semantic Audit Table:
| Cluster | Anomaly Patch % | Normal Patch % | Mean Mask Coverage |
| :---: | :---: | :---: | :---: |
| 0 | 2.507% | 97.493% | 1.638% |
| 1 | 2.479% | 97.521% | 1.616% |
| 2 | 3.156% | 96.844% | 2.184% |
| 3 | 2.518% | 97.482% | 1.647% |

- **Largest Anomaly-Share Gap**: `0.677` percentage points ($3.156\% - 2.479\%$).
- **Within-Cluster Factor Anomaly-Logit Mean Difference**: $< 0.0002$ across all 4 clusters.
- **Conclusion**: Unsupervised $k$-means partitioned predominantly normal/background patches ($> 96.8\%$ normal in every cluster), failing to form anomaly-relevant semantic responsibilities.

---

## E. Next Planned Objective: Mask-Supervised Patch Strata

To resolve the objective failure while preserving all working P1-v8 structural components:

1. **Four Explicit Patch Strata**:
   - **Strata 0**: Clean Normal Patches (GT mask $= 0$, distant from boundary).
   - **Strata 1**: Hard-Negative / Outside Context Patches (normal patches near anomaly boundary).
   - **Strata 2**: Boundary / Transition Patches (GT mask boundary contour).
   - **Strata 3**: Anomaly Core Patches (GT mask $= 1$ interior).
2. **Supervision Requirements**:
   - Soft dense routing targets aligned to patch strata.
   - Factor-specific task supervision forcing each factor head to specialize on one specific patch stratum.
   - Balanced strata sampling during batch loss computation.
   - Calibrated auxiliary loss weighting ($\le 5\%$ actual total loss contribution).
3. **Execution Guardrails**:
   - FOFS & visual experts remain `OFF`.
   - Top-K prediction remains `OFF`.
   - No $\rho$ scaling increase before objective efficacy is proven.
