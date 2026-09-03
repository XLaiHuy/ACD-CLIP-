# Seed-1 H/A E15 training-validity audit

Status: `FAIL_HARD_TRAINING_VALIDITY`.

The shared seed-1 E1 checkpoint is at global step 359. H and A each cover E2–E15 exactly once and have zero nonfinite-loss skips. H has logged three nonfinite-gradient skips (E2, E8, E9), while A has two (E2, E10). Their final global steps are therefore 5410 and 5411, respectively.

The final H15 and A15 checkpoints contain the required full-state fields and have finite model/reference tensors. The mismatch is fully accounted for by the recorded skip counts; there is no evidence of hidden steps, repeated epochs, or target-guided retraining. Because the frozen validity rule requires equal successful H/A optimizer-step totals, seed 1 is not eligible for Medical or MVTec evaluation. No target result was consulted and no restart was performed.
