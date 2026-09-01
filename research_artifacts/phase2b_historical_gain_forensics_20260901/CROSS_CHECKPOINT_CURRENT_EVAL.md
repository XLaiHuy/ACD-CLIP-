# Cross-checkpoint current-evaluator comparison

Under the current evaluator implementation, the H2 E10 replay is 90.9222 / 40.3731 Pixel AUROC/AP, H1 E9 is 90.7119 / 39.8490, and C2 P E10 is 87.9118 / 31.8093. The table keeps per-domain rows and records config/checkpoint/evaluator provenance. H1/H2 use the legacy metadata-only bypass; C2 E10 is a model-state replay based on its preserved original SHA because its physical full payload was overwritten in the disclosed serialization incident.
