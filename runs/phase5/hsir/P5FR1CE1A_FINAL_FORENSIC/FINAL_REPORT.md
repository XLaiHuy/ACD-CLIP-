# P5FR1CE1A final forensic report

## Certification

OPERATIONAL_STATUS=FORENSIC_COMPLETE_NO_SCIENTIFIC_RECONCILIATION
BASE_RESULT_COMMIT=2ef784ff91b91e3b2c2c880dfaa74c02e94445d2
FORENSIC_PROTOCOL_COMMIT=f9a2b83753e577a60854ac21e03626bda775c22e
FORENSIC_AUDIT_COMMIT=the enclosing `phase5: audit P5FR1CE1 architecture and measurement` commit
RECONCILIATION_COMMIT=NONE
REMOTE_HEAD=reported in handoff after normal push verification
MODEL_FORWARDS=0
NEW_IMAGE_READS=0
NEW_MASK_PIXEL_READS=0
TRAINING_STEPS=0
MEDICAL=false

FINAL_CERTIFICATION=SCIENTIFIC_RESULT_NOT_CERTIFIABLE
ZERO_TUNE_CERTIFICATION=CERTIFIED
OOF_CERTIFICATION=NOT_CERTIFIABLE
SCIENTIFIC_TERMINAL=NO_FOUR_FAMILY_METHOD_FULLY_SUPPORTED_HISTORICAL_RESULT_NOT_CERTIFIABLE_UNDER_AMBIGUOUS_SELECTION_SEMANTICS

## Audit statuses

ARCHITECTURE_STATUS=PASS
DEPLOYMENT_PARITY_STATUS=PASS
H6_STATUS=ENABLED_BUT_INACTIVE_FOR_PHASE2B
D_RANK_STATUS=PASS
B1_STATUS=PASS
COMMON_GEOMETRY_STATUS=PASS
PCRR_IMPLEMENTATION=PASS
CSRC_IMPLEMENTATION=PASS
ASR_IMPLEMENTATION=PASS
PGM_IMPLEMENTATION=PASS
PERCENTILE_DOMAIN_STATUS=PASS_ALL_PATCHES_THEN_INVALID_ZERO
MEASUREMENT_STATUS=PASS
MATCHING_STATUS=PASS
TRIAGE_STATUS=PASS
SHIFT_STATUS=PASS_PER_IMAGE_EVIDENCE_ONLY
OOF_ISOLATION_STATUS=PASS
CONFIG_SELECTION_STATUS=PROTOCOL_AMBIGUOUS_RESULT_AFFECTING
BOOTSTRAP_STATUS=PASS_FIXED_SEEDS_5101_TO_5107
SIGN_FLIP_STATUS=PASS_EXACT_ONE_SIDED_32768_ASSIGNMENTS
HOLM_STATUS=PASS
WINNER_LOGIC_STATUS=PASS_NO_ELIGIBLE_FAMILY
OUTPUT_CHECKER_STATUS=PASS
GT_PROVENANCE_STATUS=S1_BOOKKEEPING_DEFECT_DOCUMENTED_HISTORICAL_FAILED_ATTEMPT_MASK_FREE
RESEARCH_SCORE_STATUS=HISTORICAL_SCORE_PRESERVED_FORENSIC_INTERPRETATION_NOT_CERTIFIED
DEFECTS_S1=S1-PROV-001
DEFECTS_S2=S2-RESEARCH-001,S2-RESEARCH-002
DEFECTS_S3=S3-SELECT-001,S3-CHECK-001
DEFECTS_S4=NONE
DEFECTS_S5=NONE

HISTORICAL_EMPIRICAL_RANKING=PGM > PCRR > ASR > CSRC
HISTORICAL_FULLY_ELIGIBLE=NONE
HISTORICAL_WINNER=NONE
SELECTIONS_CHANGED_TOTAL=2 (CSRC FOLD_1; ASR FOLD_2 in ordinal sensitivity)

## Full family contract

### PCRR
- zero-tune config: `pcrr_witness_local_mean_mean`; matched win 0.671687 [0.614059, 0.727298]; delta vs B1 0.043713 [0.003528, 0.087292]; aligned-shifted 0.211391 [0.158111, 0.265575]; C_AP -0.002243 [-0.008458, 0.003110]; R_pos 0.013889 [0.003825, 0.024242]; R_neg -0.018601 [-0.027297, -0.010450].
- historical OOF matched win: 0.667476 [0.612919, 0.720437]; B1 matched win: 0.627975 [0.550413, 0.700152]; delta vs B1: 0.039502 [-0.004733, 0.090019]; aligned-shifted: 0.208268 [0.156585, 0.260019]; C_AP: -0.000964 [-0.008484, 0.005941]; R_pos: 0.013753 [0.003219, 0.024429]; R_neg: -0.018751 [-0.027953, -0.010833].
- G1=True G2=False G3=True G4=False; raw sign-flip p=0.068725586; Holm p=0.249023438.
- fold configs (historical min-max interpretation): FOLD_0=pcrr_witness_local_mean_median, FOLD_1=pcrr_pooled_peer_pairs_mean_mean, FOLD_2=pcrr_pooled_peer_pairs_mean_mean, FOLD_3=pcrr_pooled_peer_pairs_mean_mean, FOLD_4=pcrr_witness_local_mean_median

### CSRC
- zero-tune config: `csrc_spearman_average_tie_all_three_mean`; matched win 0.512130 [0.481909, 0.547055]; delta vs B1 -0.115844 [-0.202722, -0.028565]; aligned-shifted 0.013697 [-0.018107, 0.056519]; C_AP -0.022846 [-0.034360, -0.012762]; R_pos -0.008095 [-0.023346, 0.006326]; R_neg -0.036889 [-0.054946, -0.021879].
- historical OOF matched win: 0.499968 [0.485194, 0.516282]; B1 matched win: 0.627975 [0.550413, 0.700152]; delta vs B1: -0.128007 [-0.209762, -0.044549]; aligned-shifted: 0.007886 [-0.006502, 0.023804]; C_AP: -0.023146 [-0.034037, -0.012788]; R_pos: -0.012137 [-0.028047, 0.004139]; R_neg: -0.033957 [-0.051194, -0.020784].
- G1=False G2=False G3=False G4=False; raw sign-flip p=0.994873047; Holm p=1.000000000.
- fold configs (historical min-max interpretation): FOLD_0=csrc_spearman_average_tie_adjacent_max, FOLD_1=csrc_spearman_average_tie_all_three_max, FOLD_2=csrc_spearman_average_tie_adjacent_max, FOLD_3=csrc_spearman_average_tie_adjacent_max, FOLD_4=csrc_spearman_average_tie_adjacent_max

### ASR
- zero-tune config: `asr_machine_rank_mean`; matched win 0.604951 [0.555041, 0.652102]; delta vs B1 -0.023023 [-0.065996, 0.014128]; aligned-shifted 0.117202 [0.065176, 0.169946]; C_AP -0.010854 [-0.019625, -0.003671]; R_pos -0.000524 [-0.012892, 0.012039]; R_neg -0.026354 [-0.038393, -0.015563].
- historical OOF matched win: 0.608074 [0.564462, 0.650823]; B1 matched win: 0.627975 [0.550413, 0.700152]; delta vs B1: -0.019901 [-0.061237, 0.017112]; aligned-shifted: 0.113078 [0.066036, 0.157393]; C_AP: -0.011721 [-0.019548, -0.004295]; R_pos: -0.002108 [-0.016563, 0.011073]; R_neg: -0.025519 [-0.036016, -0.016656].
- G1=True G2=False G3=True G4=False; raw sign-flip p=0.817077637; Holm p=1.000000000.
- fold configs (historical min-max interpretation): FOLD_0=asr_energy_95_mean, FOLD_1=asr_energy_95_median, FOLD_2=asr_energy_99_median, FOLD_3=asr_energy_95_median, FOLD_4=asr_energy_95_median

### PGM
- zero-tune config: `pgm_sum_whitened_mean`; matched win 0.674022 [0.613286, 0.734492]; delta vs B1 0.046048 [-0.005011, 0.101467]; aligned-shifted 0.215457 [0.155545, 0.274829]; C_AP 0.002873 [-0.004115, 0.010031]; R_pos 0.015435 [0.003143, 0.027410]; R_neg -0.013078 [-0.019470, -0.007497].
- historical OOF matched win: 0.673324 [0.613195, 0.731519]; B1 matched win: 0.627975 [0.550413, 0.700152]; delta vs B1: 0.045350 [-0.003272, 0.102869]; aligned-shifted: 0.214076 [0.155707, 0.272627]; C_AP: 0.002594 [-0.004177, 0.009745]; R_pos: 0.014688 [0.002309, 0.027492]; R_neg: -0.013341 [-0.019750, -0.007811].
- G1=True G2=False G3=True G4=False; raw sign-flip p=0.062255859; Holm p=0.249023438.
- fold configs (historical min-max interpretation): FOLD_0=pgm_max_whitened_mean, FOLD_1=pgm_sum_whitened_mean, FOLD_2=pgm_max_whitened_mean, FOLD_3=pgm_sum_whitened_mean, FOLD_4=pgm_max_whitened_mean

No official reconciled metric table, selection, ranking, or winner is created because the pre-GT selection semantics are ambiguous and the counterfactual changes two selections. The min-max and ordinal sensitivity outputs are preserved in `SELECTION_SEMANTIC_SENSITIVITY.json`.

ROBUST_EMPIRICAL_RANKING=PGM > PCRR > ASR > CSRC under both tested interpretations
ROBUST_RESEARCH_INTERPRETATION=Historical PCRR-first score is not certified; qualitative design value/stability only
FULLY_ELIGIBLE_FAMILIES=NONE
PROVISIONAL_WINNER=NONE
FINAL_SELECTED_CONFIG=NONE

## Provenance

METADATA_LABELS_FIRST_READ=historical failed evaluator invocation, `load_metadata()` before `integrity_subchecks()`
MASK_PATH_METADATA_FIRST_READ=same metadata read; paths only
MASK_PIXELS_FIRST_READ=not reached in failed invocation
SCIENTIFIC_METRICS_FIRST_COMPUTED=not reached in failed invocation
GT_READ_DURING_HISTORICAL_FAILED_P5FR1C=false; MASK_READ=false; SCIENTIFIC_METRICS_READ=false; MODEL_FORWARDS=0

## Answers to the ten questions

1. The original result measured the frozen Phase2B native-logit/deployed-score pipeline and the frozen posthoc metrics as implemented. Static architecture and deployment parity pass, including inactive H6. The historical failure bookkeeping is weak for a hypothetical later crash, but the actual failed attempt was before mask pixels and metrics.
2. The audited PCRR, CSRC, ASR, PGM, B1, risk, matching, triage, shift, bootstrap, sign-flip, Holm, gates, and winner formulas pass. The research score has S2 defects. Selection semantics are ambiguous, not proven incorrectly by the pre-GT record.
3. Yes. The ordinal sensitivity changes CSRC FOLD_1 and ASR FOLD_2; PCRR and PGM do not change.
4. No. PGM > PCRR > ASR > CSRC under both interpretations, but this is not decisive evidence of a winner.
5. No family changes from eligible to eligible; all remain ineligible. Some class counts/intervals move for CSRC and ASR, but no G1-G4 eligibility outcome changes.
6. PGM is the strongest observed empirical family, not a statistically supported winner: its delta mean is positive but CI crosses zero, G2/G4 fail, raw p is 0.062255859375, and Holm p is 0.2490234375.
7. PCRR is first only under the historical research-value score. That score is not a valid certified robustness measure because of the S2 semantics/direction defects.
8. CSRC is safely closable as a current research direction for this frozen study: it is last under both interpretations and fails all usefulness gates. That does not prove the underlying idea impossible in general.
9. No. This result is not a certified foundation for the next phase. It is a bounded forensic result with no eligible family and an unresolved selection semantic that requires a fresh preregistered study.
10. The strongest defensible next hypothesis is qualitative: parameter-free peer-geometry evidence, especially PGM/PCRR, may carry alignment-grounded signal, but any next study must first preregister an unambiguous ordinal/min-max selection rule and a corrected research-value rubric, then validate on untouched data. No next phase is started here.
