# H2 DFG and prompt-branch forensics

## DFG

The existing DFG diagnostics show finite, normalized, non-uniform stage
weights. Final stage-3 abnormal weights are:

| run | weights | entropy |
|---|---|---:|
| Seed 1 H | `[.0437,.3382,.6181]` | `.8008` |
| Seed 2 H | `[.0337,.3429,.6234]` | `.7757` |
| Seed 1 A | `[.1135,.5965,.2900]` | `.9142` |
| Seed 2 A | `[.1423,.6130,.2447]` | `.9220` |

The stage-weight sum error is at most approximately `1.2e-7`. A's middle
stage preference is consistent across seeds, while H's third-stage preference
is consistent. This rules out NaN/inf DFG weights or a trivially uniform gate
in the archived traces. It does not establish that the selected stages are
correct on target images: no target-conditioned spatial/group trace is
stored.

`DFG_BOTTLENECK=POSSIBLE_BUT_NOT_DEMONSTRATED`.

## Prompt branch

The hybrid schedule is deterministic in all four logs: alpha is zero while
the soft prompt is frozen, prompt training begins at the prescribed unfreeze
epoch, and alpha reaches `.2` by E15. The KG term declines while the soft
branch moves away from the hard branch. At E15, the logs show nonzero context
gradient statistics after unfreezing and decreasing normal/abnormal soft/hard
cosines in parts of the trajectory. This is evidence of prompt drift in the
source run, but no target margin or target text-feature trace exists.

`TEXT_BRANCH_BOTTLENECK=POSSIBLE_BUT_NOT_CAUSALLY_IDENTIFIED`.

The public protocol does not specify enough prompt construction detail to
prove external equivalence. The current H/A contrast, however, keeps the
prompt schedule matched internally.

## Anchor interaction

Family telemetry shows Q/K and SS2D are the main locations with active Anchor
budget, while LoRA is mostly Anchor-negligible. The global effective ratio is
tiny, so the evidence does not support a claim that the prompt/DFG branch was
globally over-constrained by Anchor. A source-only numerical trace is needed
before changing either branch.
