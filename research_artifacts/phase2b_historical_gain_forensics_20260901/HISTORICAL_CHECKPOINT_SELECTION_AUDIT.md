# Historical checkpoint-selection audit

H1 E9 and H2 E10 are retrospective champions. H2’s run context explicitly selects E10 by the six-Medical pixel AP mean; H1 is documented as the Phase1 best checkpoint with Medical results. These are RETROSPECTIVE_BEST, not clean target-blind baselines. They remain valid for forensic recovery and understanding lost performance, but their historical winning-epoch scores must not be presented as target-blind prospective results.

This audit did not use new Medical evaluation to choose the H1/H2 replay checkpoint. The current C2 E10 decomposition is fixed for the comparison. Any new parent/anchor training must freeze a source-only or fixed-epoch rule before reading new Medical results.
