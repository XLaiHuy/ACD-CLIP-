# Predecessor CIR-control ingest

The existing PA control archive was ingested without rerunning training or evaluation. PA is a no-CIR, image-anchor control using the canonical parent contract. Its final decision is CIR_TRAINING_VALUE=INCONCLUSIVE, INFERENCE_RMT_VALUE=NEUTRAL, and FINAL_ARCHITECTURE=MIXED_UNRESOLVED; its source and Medical status are PASS and no target tuning occurred in that run.

The factorial evidence shows that anchor/no-CIR (PA) improves C2 P on Medical pixel AUROC/AP at most epochs, while CIR-with-anchor versus PA is mixed by metric and epoch. The conditional alpha comparison is near neutral, so new strong-parent candidates default to native inference alpha=0. This predecessor evidence does not independently justify a full CIR trajectory; a CIR trajectory would require a bounded strong-parent source-only benefit.

Exact derived comparisons (A0-PA, PA-P, A0-C_OLD, and factorial interaction) are in PREDECESSOR_CIR_CONTROL_INGEST.json and reference the frozen PA CSVs.
