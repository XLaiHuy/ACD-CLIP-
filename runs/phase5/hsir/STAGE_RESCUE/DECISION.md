DECISION: CONSENSUS_DILUTION_SUPPORTED
INPUT INTEGRITY: PASS; exact Phase5-A.1 provenance and frozen predictor semantics verified
OUTPUT INTEGRITY: PASS; final AP/AUROC evaluator parity and class-level checks passed

FINAL_DEPLOYED_CONSENSUS:
  final_AP_class_macro_mean: 0.452476151
  final_AUROC_class_macro_mean: 0.934641042

FIXED_STAGE_COUNTERFACTUALS:
  stage8_AP_class_macro_mean: 0.323734034; stage8_AUROC_class_macro_mean: 0.958180954; stage8_AP_delta_class_macro_mean: -0.128742117
  stage16_AP_class_macro_mean: 0.453559019; stage16_AUROC_class_macro_mean: 0.964648453; stage16_AP_delta_class_macro_mean: 0.001082867
  stage24_AP_class_macro_mean: 0.367586912; stage24_AUROC_class_macro_mean: 0.893434992; stage24_AP_delta_class_macro_mean: -0.084889239

SELECTED_POSITIVE_RESCUE:
  selected_positive_G_rescue_mean_class_median: 0.001236406
  selected_positive_G_rescue_median_class_median: -0.002859309
  G_rescue_vs_C_AP_class_median: 0.21617881628955735
  G_rescue_vs_R_pos_class_median: 0.2980228359886886
  internal_stage_rescue_delta_AP_class_macro_mean: 0.009175164
  A1_positive_only_oracle_delta_AP_class_macro_mean: 0.104536716
  fraction_of_A1_oracle_recovered_class_median: 0.12075524902719677

STAGE_IDENTITY:
  pixel_winner_fractions: {'stage8': 0.6313662674287378, 'stage16': 0.2906640212603298, 'stage24': 0.07796971131093232}
  class_winner_counts: {'stage8': 8, 'stage16': 3, 'stage24': 1}

NORMAL_SAFETY:
  all_normal_inflation_mean_class_median: 0.002704714402037331
  harmful_normal_inflation_mean_class_median: 0.0035994958205940453

NEXT_BRANCH: B1
