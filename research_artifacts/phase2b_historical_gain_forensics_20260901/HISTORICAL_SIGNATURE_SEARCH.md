# Historical signature search

The repository and run were found by exact signature search, not by guessing from metric values.

- H1 signature phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3 -> /home/ai4/caohuy/ACD-CLIP-base-new-phase1/phase1_best_checkpoints/phase1b_v3c_fp32attn_final/e09_best_final_anchor_adapter.pth; source snapshot 0232200e964c02c328eec09dbe842f327f72fcd9.
- H2 signature phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch -> /home/ai4/caohuy/ACD-CLIP-base-new-phase1/runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch; exact implementation commit e03966997d4cecfd985943a4053a93e1e40197ec.
- H2 checkpoint SHA ae27443f99020588298a9ecc6dfc833a83ebe7a752f00e8524042d5a84a2c0cb and historical evaluator SHA 7bdd8cc6ada90467285a79ced9599ed778c6dc2a0ba6596d2f3311fa637fae9d were verified.
- C2 corrected parent artifacts are under research_artifacts/cir_rmt_v2/corrective_matched_retrain_20260830/; the original E10 SHA is preserved as 31ca8344c646693d0ee51941d39f28aa07b6a102c49d1efdc5e3cdf2ec8bcc50.

The H2 run began after commit e03966997d4cecfd985943a4053a93e1e40197ec and before later result-document commits. The recovered log, code, checkpoint, and result chronology identify the exact historical implementation.
