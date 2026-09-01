# H1/H2/C2 full contract matrix

The CSV is the authoritative cell-by-cell matrix. Each value has an evidence level and source. RECOVERED and DERIVED cells are evidence-backed or arithmetic; UNKNOWN is retained where the historical artifact did not expose a setting.

H2 and C2 share the core CLIP/DFG/Adam/base-LR/StepLR/batch structure, but differ in K-reg, KG coefficient, AMP versus FP32, prompt LR, horizon/candidate selection, and evaluator. H1 is a hard-prompt V3c parent, so it is a useful historical comparator but not an architecture-identical H2 control.
