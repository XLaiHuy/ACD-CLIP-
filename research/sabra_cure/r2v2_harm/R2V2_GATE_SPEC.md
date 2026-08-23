# R2-v2 Gate Specification

Mandatory gates: audit PASS; macro realized coverage >=10%; accepted wrong-sign
rate <=5%; weighted accepted-harm density reduction >=25% versus unfiltered
direction; macro pAP strictly above native and published R2; at least 9/12
classes non-regressing pAP versus native; macro pAUROC delta >=-0.005, reusing
the exact R2 guardrail. Harm-aware versus binary is reported only: if binary
clearly dominates, report `HARM_WEIGHTING_NOT_SUPPORTED`; it is not silently
promoted. Any mandatory failure is `R2V2_SCIENTIFIC_STOP`; no R3/R4 follows.
