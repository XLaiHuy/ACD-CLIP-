# Phase2B historical-to-current gap decomposition

The decomposition is:

historical H2 through historical evaluator
  -> same H2 checkpoint through current evaluator
  -> current C2 parent E10 through current evaluator.

At the six-domain pixel macro this is 90.9750 -> 90.9222 -> 87.9118 AUROC and 40.3483 -> 40.3731 -> 31.8093 AP. The evaluator components are -0.0528/+0.0247 points; the residual checkpoint/training/config component is -3.0104/-8.5637 points. The total gap is -3.0632/-8.5390 points.

The residual is deliberately not called a pure training effect. It includes the H2-to-C2 K-reg removal, KG coefficient change, AMP-to-FP32 migration, prompt LR change, horizon/candidate/selection changes, possible loader differences, and checkpoint-provenance limitations. The old CIR missing scheduler.step is a separate confirmed bug, not the explanation of this H2-to-C2 decomposition.
