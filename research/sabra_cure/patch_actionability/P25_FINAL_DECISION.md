# P25 Final Decision

`P25_TARGET_PANEL_NO_GO`

P25 stopped before the scientific marker. The frozen target-panel contract
requires 4096 patches per class while also imposing at most 16 target patches
per image. The immutable canonical VisA inventory has 150--201 images per
class, so maximum capacities are only 2400--3216 patches per class. The total
capacity is 34,592, below the required 49,152 by 14,560.

This is a mathematical feasibility contradiction, not a scientific result:
deterministic stratum redistribution cannot exceed a per-image cap. Raising the
cap, lowering the quota, dropping classes, or otherwise changing the panel
would alter the preregistered P25 protocol and is forbidden.

No P25 target V value, model fit, Q1/Q2 fold, target policy, or attempt marker
was created. MVTec/Medical reads, additional CLIP forwards, and Phase2B steps
remain zero. The learned patch-actionability branch is not scientifically
closed by this pre-marker no-go; any revised panel requires explicit user
review and a new preregistration.
