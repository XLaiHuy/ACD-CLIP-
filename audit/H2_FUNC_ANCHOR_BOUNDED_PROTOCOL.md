# H2 bounded functional feature-anchor protocol

This source-only test compares matched `A_SHORT` and `A_FUNC_SHORT` arms from
the same frozen shared-E1 checkpoint. Both use the historical mixed FP16/FP32
protocol, Safe Anchor, optimizer, source batches, prompt schedule, DFG/SS2D,
and 500 maximum optimizer-step attempts. The candidate alone adds equal-weight
stage-2/stage-3 normalized native-token cosine loss against a frozen E1 teacher.

At exact E1 equality this cosine loss has zero gradient. The preregistered
eight-batch calibration therefore uses a discarded task-only shadow trajectory:
after each ordinary source task update it measures the task and unweighted
functional gradients, sets one lambda for a 10% median auxiliary ratio, then
resets both experimental arms to the untouched shared E1. No metrics or target
data participate in calibration.
