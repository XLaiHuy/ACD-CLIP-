# Iteration A Report: Real Data Audit

## Exact Source State
The codebase uses P1-v8 structural checkpoint `adapter_3.pth`. `h6_expert_enabled` is False. The dense patch router operates unsupervised.

## Confirmed & Rejected Hypotheses
- **H13 (clean-normal and anomaly-image-outside factors collapse)**: `CONFIRMED`.
- **All other hypotheses**: `INSUFFICIENT_EVIDENCE` (stopped iteration early due to H13 blocking).

## Files Inspected
- `tools/audit_p1_v8_2.py`
- `model/h6/router.py`
- `model/h6/losses.py`
- `model/h6/model.py`
- `train.py`

## Files Modified
- `tools/audit_p1_v8_2.py`

## Artifacts Produced
- `runs/phase4/p1_v8_2_iteration/A_real_data_audit/ROLE_SPECIALIZATION_AUDIT.md`
- `runs/phase4/p1_v8_2_iteration/EVIDENCE_LEDGER.md`
- `runs/phase4/p1_v8_2_iteration/FINAL_ITERATION_REPORT.md`

## Active Background Processes
None.

## Git Status
 .../adapter_10.pth                                 |  Bin 133 -> 56452037 bytes
 .../test.log                                       |  201 ++++
 .../train/exact_results_Brain_test_epoch_12.csv    |    3 -
 .../train/exact_results_Brain_val_epoch_12.csv     |    3 -
 .../exact_results_Colon_Kvasir_test_epoch_12.csv   |    3 -
 .../exact_results_Colon_Kvasir_val_epoch_12.csv    |    3 -
 .../exact_results_Colon_clinicDB_test_epoch_12.csv |    3 -
 .../exact_results_Colon_clinicDB_val_epoch_12.csv  |    3 -
 .../exact_results_Colon_colonDB_test_epoch_12.csv  |    3 -
 .../exact_results_Colon_colonDB_val_epoch_12.csv   |    3 -
 .../train/exact_results_Liver_test_epoch_12.csv    |    3 -
 .../train/exact_results_Liver_val_epoch_12.csv     |    3 -
 .../train/exact_results_Retina_test_epoch_12.csv   |    3 -
 .../train/exact_results_Retina_val_epoch_12.csv    |    3 -
 .../train/medical_val_results_by_dataset.csv       |   12 +-
 .../train/medical_val_results_by_epoch.csv         |    2 +-
 .../train/medical_validation_selection.json        |   28 +-
 .../progress1_v7_full_seed0_ready3/train/test.log  | 1079 ++++++++++++++++++++
 18 files changed, 1301 insertions(+), 57 deletions(-)
## Git Status
 M runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/adapter_10.pth
 M runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/test.log
 D runs/phase4/progress1_v7_full_seed0_ready3/train/exact_results_Brain_test_epoch_12.csv
 D runs/phase4/progress1_v7_full_seed0_ready3/train/exact_results_Brain_val_epoch_12.csv
 D runs/phase4/progress1_v7_full_seed0_ready3/train/exact_results_Colon_Kvasir_test_epoch_12.csv
 D runs/phase4/progress1_v7_full_seed0_ready3/train/exact_results_Colon_Kvasir_val_epoch_12.csv
 D runs/phase4/progress1_v7_full_seed0_ready3/train/exact_results_Colon_clinicDB_test_epoch_12.csv
 D runs/phase4/progress1_v7_full_seed0_ready3/train/exact_results_Colon_clinicDB_val_epoch_12.csv
 D runs/phase4/progress1_v7_full_seed0_ready3/train/exact_results_Colon_colonDB_test_epoch_12.csv
 D runs/phase4/progress1_v7_full_seed0_ready3/train/exact_results_Colon_colonDB_val_epoch_12.csv
 D runs/phase4/progress1_v7_full_seed0_ready3/train/exact_results_Liver_test_epoch_12.csv
 D runs/phase4/progress1_v7_full_seed0_ready3/train/exact_results_Liver_val_epoch_12.csv
 D runs/phase4/progress1_v7_full_seed0_ready3/train/exact_results_Retina_test_epoch_12.csv
 D runs/phase4/progress1_v7_full_seed0_ready3/train/exact_results_Retina_val_epoch_12.csv
 M runs/phase4/progress1_v7_full_seed0_ready3/train/medical_val_results_by_dataset.csv
 M runs/phase4/progress1_v7_full_seed0_ready3/train/medical_val_results_by_epoch.csv
 M runs/phase4/progress1_v7_full_seed0_ready3/train/medical_validation_selection.json
 M runs/phase4/progress1_v7_full_seed0_ready3/train/test.log
?? P1_fast_audit_runtime_decision_tree_gemini_prompt_v2_nonregression.md
?? TIER3_ROUTER_PATCH_CONTRACT.md
?? fix_getattr.py
?? generate_argparse.py
?? inject_args.py
?? inject_args_fixed.py
?? phase2b.log
?? profile.log
?? runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/B0_-_Phase2B_baseline/
?? runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/exact_results_Brain_val_epoch_10.csv
?? runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/exact_results_Colon_Kvasir_val_epoch_10.csv
?? runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/exact_results_Colon_clinicDB_val_epoch_10.csv
?? runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/exact_results_Colon_colonDB_val_epoch_10.csv
?? runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/exact_results_Liver_val_epoch_10.csv
?? runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/exact_results_Retina_val_epoch_10.csv
?? runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/medical_val_results_by_dataset.csv
?? runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/medical_val_results_by_epoch.csv
?? runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/medical_validation_selection.json
?? runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/phase2b_triage/
?? runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/protocol/
?? runs/phase4/p1_fast_audit/
?? runs/phase4/p1_v8_2_iteration/
?? runs/phase4/p1_v8_evidence/
?? runs/phase4/p1_v8_minimal_wiring_smoke/
?? runs/phase4/p1_v8_specialization_overnight/
?? runs/phase4/progress1_v7_full_seed0_ready3/train/A0_-_legacy_P1-v7/
?? runs/phase4/progress1_v7_full_seed0_ready3/train/A1_-_hard-anchor_baseline_inside_P1_checkpoint/
?? runs/phase4/progress1_v7_full_seed0_ready3/train/A2_-_target_P1-v8/
?? runs/phase4/progress1_v7_full_seed0_ready3/train/A3_-_sparse_comparison/
?? runs/phase4/progress1_v7_full_seed0_ready3/train/A4_-_optional_safety_comparison/
?? runs/phase4/progress1_v7_full_seed0_ready3/train/B0_-_Phase2B_baseline/
?? runs/phase4/progress1_v7_full_seed0_ready3/train/protocol/
?? runs/phase4/progress1_v8_structural_smoke_seed0/
?? runs/phase4/tier3_m4_seed0/
?? stage_c_ablation.py
?? temp_rewrite.py
?? test_shape.py
?? tools/audit_p1_v8_2.py
?? tools/instrument_train.py
?? tools/phase4_v8_minimal_fix.py
?? tools/probe_augmentation.py
?? train_profile.py
?? triage_A0_A3.log
?? wiring.log
## Primary Decision
FIX_OBJECTIVE_WIRING
