# P21 Gate and Routing Spec

Stage A P20 parity must reproduce native, SAFE20, EXPAND40, context, and image-oracle pAP to <=1e-12 and preserve 12-class/hash/order provenance. Failure is `P21_ENGINEERING_STOP`.

Stage B evaluates A0. If strong, Stage C is skipped and Stage D runs. If weak, Stage C evaluates exactly A1 with SAFE30. A1 weak gives `P21_CONTEXTUAL_ACTION_SPACE_INSUFFICIENT` and skips Stage D.

Stage D floors for P0/P1/P2 are median held Spearman >=.20, positive-Spearman class count >=9/12, and sign accuracy >=.60. P0 fail/P1 pass supports rank-objective mismatch; P1 fail/P2 pass supports action-impact feature gap; all fail supports `IMAGE_VALUE_NOT_GT_FREE_PREDICTABLE`; P2 aggregate success without breadth supports group-shift limitation; P2 meeting all floors justifies, but does not implement, a new low-capacity rank-controller preregistration.

Any provenance, parity, exactness, child, firewall, or post-audit mismatch is `P21_ENGINEERING_STOP`; no scientific interpretation or rerun follows.
