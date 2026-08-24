# P21 Predictability Contract

Stage D is conditional on strong A0/A1 headroom and uses the smallest strong family. Opportunity targets are native-anchored exact AP changes with every other image NATIVE; NATIVE has target zero and ties are conservative. They are GT-bearing source diagnostic labels only.

F0 is the frozen 16 P14 GT-free context features. F1 adds four GT-free native-versus-action features per active non-native action: mean signed score delta, q90 absolute delta (`linear`), top-decile membership crossing fraction, and stable-rank Spearman. F1 contains no mask, y, utility, or label.

Exactly three probes are allowed: P0 centered ridge F0/lambda1; P1 linear pairwise logistic ranker F0/lambda1; P2 same ranker F0+F1/lambda1. Scaling is source-train-only median/IQR, intercept unregularized, float64, deterministic zero initialization. Pairs are all within-source-class pairs with `abs(value_i-value_j)>EPS`; no sampling/balancing/sweep. Held class is excluded from all training/scaling/pairs.
