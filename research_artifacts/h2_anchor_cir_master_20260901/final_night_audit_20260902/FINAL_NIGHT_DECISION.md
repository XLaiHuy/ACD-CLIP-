# Final-night decision

Decision: `STOP_AUDIT_NO_TRAINING`.

This is decision-tree Case B:

1. Same-E10 Medical RA is below R in Pixel AUROC (`-0.0158400011`) while Pixel AP rises (`+0.0172562369`).
2. The historical per-tensor relative-L2 Anchor is ill-conditioned. At RA E10, `lambda_anchor * ||g_anchor|| / ||g_task|| = 40068.9185` on the fixed audit batch; at RA E16 it is `31138.6838`.
3. The source gate overlaps VisA training at `96/96`, so it is an in-distribution assessment rather than a clean holdout.
4. Historical H2 and new R do not have a complete shared seed/RNG/replay contract, and the exact new R E1 state needed for a same-trajectory reference audit is missing.

Consequences:

- Geometry/SRTR is not authorized.
- No architecture change, RMT change, FP32 switch, optimizer change, loss change, or long training was started.
- No corrected Anchor was implemented or trained tonight.
- The current RCA result cannot isolate a clean RMT gain because RCA inherits the invalid Anchor trajectory.

The next safe experiment is one preregistered same-trajectory Anchor correction using a globally normalized image-adapter distance, capped at E1-E10, after the seed/RNG and source-validation contracts are made explicit. It should be evaluated before any geometry/SRTR decision. This report does not authorize that run.
