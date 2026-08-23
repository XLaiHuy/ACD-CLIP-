# P14 Gate Specification

Mandatory gates: G1 pre/post audit PASS; G2 accepted wrong-sign <=5%; G3
weighted-harm reduction >=50%; G4 coverage >=19.34%; G5 EXPAND40 usage >=10%; G6
macro pAP delta vs native >=+.0025; G7 >=9/12 non-regressing classes; G8 >=7/12
strictly improving classes; G9 exact R2-v2 pAUROC guardrail; G10 macro pAP
strictly exceeds reconstructed SAFE20; G11 each q follows frozen OOF selection.

All pass yields `P14_PASS`; otherwise `P14_SCIENTIFIC_STOP`.  No gate may be
relaxed after results.  Required reports include native, published R2-v2
reference, SAFE20, ALWAYS_EXPAND40 ablation, primary context policy, and the
image oracle (`V_j>0`) labelled `POST_HOC_ORACLE_DIAGNOSTIC`.
