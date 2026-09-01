# Historical Phase1 repository identity

H1 is the V3c run named phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3 in /home/ai4/caohuy/ACD-CLIP-base-new-phase1. The recovered source snapshot is commit 0232200e964c02c328eec09dbe842f327f72fcd9; the run log, test log, and archived E9 checkpoint all carry the matching V3c DFG signature. The repository remote is https://github.com/XLaiHuy/ACD-CLIP-.git.

The fp32attn name refers to the DFG weight-residual/attention path, not full FP32 training. The H1 log explicitly enables AMP, and the source constructs GradScaler and autocast.
