# SABRA Trust-v2 M4 Correction Follow-up

Protocol ID: TRUST_V2_M4_CORRECTION_FOLLOWUP

Status: FROZEN

## Purpose and validity boundary

The prior Trust-v2 VisA run is invalid: TRUST_V2_STUDY_INVALID. It omitted the
preregistered M4 diagnostic and cannot be used as confirmatory evidence.

This follow-up corrects an implementation/protocol mismatch only. It does not
redesign Trust-v2. Any numerical result from the invalid prior run must not be
used to modify model definitions, thresholds, feature selection, evidence
ladders, gates, or safety rules.

The follow-up reuses the existing finalized GT-free Trust-v2 cache unchanged.
No new scientific feature, cache field, Phase2B inference, or Phase2B weight is
authorized or required.

## Frozen cache and provenance

The cache is runs/phase5/sabra/TRUST_V2_DEVELOPMENT/cache and is accepted only
when TRUST_V2_GT_FREE_MANIFEST.json remains finalized and its existing
provenance checks pass. The cache has 2,162 records, 12 classes, STRONG p16
coverage, PASS baseline parity, and PASS p16 geometry parity.

The existing cache already contains all required M0-M3 inputs and:
    D_rel = abs(PGM_baseline_rank - PCRR_baseline_rank)

The D_rel identity is checked shard-by-shard before the follow-up audit. The
cache is reused byte-for-byte and is not rebuilt.

## Frozen Trust-v2 models

M0 consists of E only.

M1 consists, in this exact order:
    E
    peer_coherence
    query_support_mean
    peer_eigen_entropy
    stage_query_profile_disagreement

M2 consists of all M1 columns followed by:
    S9
    R9

M3 consists of all M2 columns followed by:
    S16
    R16

M3 is eligible only under the already-frozen p16 coverage rule:
STRONG or ACCEPTABLE coverage as defined by the original Trust-v2 protocol.
No gate is loosened.

All M0-M3 formulas, p9/p16 definitions, PGM, preprocessing, LOCO folds,
StandardScaler training-fold isolation, logistic configuration, evidence
ladder, safety metrics, Need C1, Authority-v2, and GT semantics are unchanged.

## Deterministic non-PCRR selection

The exact previously frozen deterministic selection rule is retained:

1. Determine M1, M2, and M3 evidence classes against M0.
2. Among eligible models, choose the simplest model reaching the highest
   evidence category.
3. If multiple models are in the same category, prefer the simpler model unless
   the more complex model has at least +0.003 additional mean AUROC or a
   meaningful later Authority safety improvement.
4. M3 cannot be selected if p16 coverage is insufficient.
5. No model below WEAK_POSITIVE_EVIDENCE may be frozen for external confirmation.

The invalid prior observed results must not influence this selection rule.

## Corrected M4 diagnostic

After deterministic selection, let selected_non_PCRR_model be the selected M0,
M1, M2, or M3 feature matrix and preserve its exact column order.

Define M4 literally as:
    M4 = selected_non_PCRR_model columns, followed by D_rel

where:
    D_rel = abs(PGM_baseline_rank - PCRR_baseline_rank)

M4 is fit with the same class-held-out LOCO procedure, preprocessing, and
logistic configuration as the other diagnostics. It is PCRR incremental
diagnostic-only and is not eligible to replace the non-PCRR Trust candidate.

The primary PCRR comparison is, per held-out VisA class:
    AUROC(M4) - AUROC(selected_non_PCRR_model)

Retain D_rel only when the incremental effect has:
    positive mean
    non-negative median
    no catastrophic tail
    at least WEAK_POSITIVE_EVIDENCE

Otherwise:
    PCRR_STATUS = DROP

Forbidden PCRR alternatives:
    E + PCRR
    PGM + alpha*PCRR
    weighted fusion
    PCRR rescue model
    PCRR threshold tuning
    feature-subset search
    model selection using M4

## Firewall and completion

VisA RGB and VisA GT are allowed only after this protocol and implementation
are pushed and the GT-free cache gate passes. MVTec remains forbidden until a
candidate is eligible, fully frozen, committed, pushed, and verified. Medical
data remain forbidden permanently.

Counters remain:
    MEDICAL_READS = 0
    MVTEC_READS_BEFORE_FREEZE = 0
    PHASE2B_TRAINING_STEPS = 0
    TRUST_V2_MODEL_SELECTION_AFTER_MVTEC = 0

The follow-up must produce M0-M3 audit, deterministic selection, corrected M4,
PCRR decision, stable-but-wrong V2 report, Authority-v2, statistics,
adversarial review, and final VisA decision artifacts. Invalid prior artifacts
remain invalid evidence and must not be staged as valid results.

No MVTec access is permitted before the candidate freeze checkpoint.
