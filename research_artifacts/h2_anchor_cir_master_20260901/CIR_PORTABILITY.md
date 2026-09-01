# CIR-V2 portability audit

Status: PASS for the pre-training port.

- historical H2 group order, normal/abnormal orientation, and three-stage geometry are preserved;
- peer count K=8 and spatial radius=3 are unchanged;
- peer selection and robust delta are detached, while native DFG and score paths remain differentiable;
- configured transport direction is `abnormal_minus_normal_plus`;
- training alpha is `0.5`; deployment alpha is `0.0` through the native historical evaluator;
- exact-score-space alias uses the frozen optimized score implementation;
- fixed-input DFG/logit/probability parity is PASS with zero measured max-absolute differences;
- fixed E0 peer validity is 1.0 and synthetic source-side sign sanity passes.

CIR is train-time only in RCA. Inference alpha=.5 is not part of this master experiment, so no inference-RMT causal claim is made here.
