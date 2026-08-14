# V1.5 pre-flight decision

`V1_DIAGNOSTIC_REPAIRED`

The historical V1 label `max_relative_correction` referred to the maximum
active-training value, whereas `mean_relative_correction` referred to final
evaluation images. They are different populations, so a smaller reported
maximum is not a numerical contradiction; V1.5 records the definitions
explicitly.

No V1 variant checkpoint was retained. Same-checkpoint ACTIVE/OFF/ZERO_DELTA
attribution is therefore not reconstructible without inventing state. The
authorized route is Branch E: one fresh shared OpenAI-CLIP warmup, paired BASE
and current-V1 continuations, 32 active microbatches, and retained compact
checkpoints for the prescribed causal decomposition.
