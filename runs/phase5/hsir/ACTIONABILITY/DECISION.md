DECISION: RANK_RISK_POSITIVE_SIDE_ONLY
INPUT INTEGRITY: PASS; authoritative pixel cache absent; exactly one fresh inference exposure pass
TARGET PARITY: PASS; mean C_AP and both pairwise inversion identities match frozen Phase5-A definitions

D_RANK:
  selected_positive_fraction_at_20pct_median: 0.001087
  positive_AP_damage_capture_at_20pct_median: 0.101728
  positive_pairwise_risk_capture_at_20pct_median: 0.319194
  negative_pairwise_risk_capture_at_20pct_median: 0.187225
  both_oracle_AP_delta_median: 0.078899

U_CONF:
  selected_positive_fraction_at_20pct_median: 0.014307
  positive_AP_damage_capture_at_20pct_median: 0.885687
  positive_pairwise_risk_capture_at_20pct_median: 0.251414
  negative_pairwise_risk_capture_at_20pct_median: 0.627387
  both_oracle_AP_delta_median: 0.423106

D_LOGIT:
  selected_positive_fraction_at_20pct_median: 0.013604
  positive_AP_damage_capture_at_20pct_median: 0.864022
  positive_pairwise_risk_capture_at_20pct_median: 0.355487
  negative_pairwise_risk_capture_at_20pct_median: 0.577858
  both_oracle_AP_delta_median: 0.408451

COMPLEMENTARITY:
  union_actual_coverage_median: 0.190391
  union_oracle_AP_delta_median: 0.417955
  matched_D_rank_oracle_AP_delta_median: 0.076668
  matched_U_conf_oracle_AP_delta_median: 0.420791

Q1_why_D_rank_damage_capture_high: D_rank selects a median positive fraction of 0.001087, but its selected positive pixels capture 0.319194 of positive pairwise risk; negative pairwise-risk capture is 0.187225. The concentration is therefore risk-specific rather than a large anomaly-pixel share.
Q2_why_D_rank_both_oracle_lower_than_U_conf: The D_rank BOTH-oracle AP delta median is 0.078899, versus 0.423106 for U_conf; D_rank identifies ranking-risk mass that is not equivalent to the pixels with maximum score-repair leverage.
Q3_D_rank_failure_mode: D_rank is primarily weak-anomaly-positive actionability: positive-only oracle delta exceeds negative-only delta in 12 of 12 classes, with median deltas 0.076967 versus 0.002762.
Q4_U_conf_complementarity: U_conf identifies a distinct harmful-Normal/high-repair mode: rank/confidence Jaccard is 0.113720, but the matched-budget union beats both single-selector controls in only 6 of 12 classes.
Q5_matched_budget_union: The fixed top-10% D_rank UNION U_conf beats both matched-budget single-selector controls in 6 of 12 classes.
