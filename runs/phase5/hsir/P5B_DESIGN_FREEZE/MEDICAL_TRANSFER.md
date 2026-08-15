# Medical-transfer rationale and limits

## Transfer-compatible primitives

A future candidate should rely only on relative within-image quantities that do not name an industrial class or defect type:

- same-image reference relations from `E_nonlocal`;
- within-image Phase2B rank positions or percentiles;
- relative stage disagreement `D_rank` as risk, never anomaly evidence;
- feature similarity/distance relations from the frozen encoder;
- explicit abstention when references are invalid or pair evidence is ambiguous.

These are transfer-compatible design preferences, not evidence of medical validity.

## Non-transferable or unverified parts

The current evidence does not justify transferring VisA constants or assumptions to medical data. K=8, top-20% risk eligibility, 10x10 cells, the shifted control, any rank/score/spatial cutoff, and any action budget remain protocol-specific until separately justified. Industrial anomaly classes, object texture, image resolution, and defect prevalence differ from medical imaging. No medical images, labels, tuning, or external frozen test were used.

A future medical test must freeze the candidate before medical outcomes, preserve the same GT-free inference/action firewall, and report abstention and safety separately from anomaly performance. CLIP transfer, WinCLIP/AnomalyCLIP, and MedCLIP support the need for domain-specific validation; they do not prove that a post-hoc native score projection transfers.

## Status

Medical transfer is **unproven**. The design is not ready for a single full VisA candidate evaluation, and no medical experiment is authorized by this freeze.
