# V1.5 causal decision

`V1_HARM_DOMINATED_BY_OPTIMIZATION_DRIFT`

The historical V1 label `max_relative_correction` referred to active-training
microbatches, while `mean_relative_correction` referred to final evaluation
images. Those are different populations, so their order is not a numerical
contradiction.

The fresh shared-warmup paired replay retained adapter-only checkpoints for
same-checkpoint inference. ACTIVE minus OFF was negligible, while OFF versus
paired BASE explained essentially 100% of AP (`-0.010885`) and AUROC
(`-0.026273`) degradation, with exact causal reconstruction. The only
authorized recovery is Branch A: gradient-isolated conditional correction.
