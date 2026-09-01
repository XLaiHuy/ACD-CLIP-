# Historical Phase1 H1 contract

H1 is hard-prompt V3c with ViT-L-14-336, image size 518, three groups, attention DFG dimension 256/tau 8, SS2D weight residual gamma .2, beta warmup010 to .1, Adam base image LR 1e-3, AMP/autocast/GradScaler enabled, and StepLR gamma .9 stepped after the epoch before save. H1 has no hybrid soft prompt and no K-reg.

The historical evaluator uses pixel stride 4 and the same legacy image score construction used by H2. The E9 checkpoint is retrospective best rather than a source-only prospective selection. Full details are in HISTORICAL_PHASE1_CONTRACT.json.
