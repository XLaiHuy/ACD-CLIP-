# P5-E0R Frozen HRIP Evaluation Recovery

Original P5-E0 remains historically `P5E0_HRIP_AUDIT_INVALID`; this separate E0R audit does not rewrite it.

P5-E0R recovered the preregistered evaluation of the frozen P5-E0 evidence with zero model forwards, zero training, and no medical evaluation. No frozen HRIP evidence was changed or recomputed. The only semantic correction was application of the already-preregistered `tools/audit_phase5_hsir.py::shifted_map` independently to each image's 518x518 HRIP map before class concatenation.

G1=True, G2=False, G3=True, G4=False. The recovered scientific terminal is `NOT_REACHED`. E0R terminal is `P5E0R_FROZEN_EVIDENCE_INVALID`. G1-G4 use the original frozen protocol, matching, risk population, triage budget, bootstrap repetitions, and seeds. Candidate remains `NONE`; E1 is not implemented.

High HRIP means that the peer-supported Normal-ish reconstruction poorly explains the query. It does not mean anomaly is confirmed.
