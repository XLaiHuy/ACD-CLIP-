# Same-checkpoint cross-evaluator result

The exact H2 E10 model-state checkpoint was evaluated through both the historical and current evaluator paths. At the six-domain pixel macro, historical H2 is 90.9750 AUROC / 40.3483 AP and current evaluator H2 is 90.9222 / 40.3731. The evaluator component is -0.0528 / +0.0247 percentage points. Supported three-domain image metrics shift by -0.0207 / +0.0107 points.

This bounded result shows evaluator migration is not the dominant explanation of the H2-to-C2 loss. Historical output is rounded to two decimal percentage points, and legacy checkpoint metadata identity was bypassed only for loading; weights were not changed.
