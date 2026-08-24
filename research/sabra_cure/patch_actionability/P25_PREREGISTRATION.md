# P25 — Patch-Level Benefit / Actionability Identifiability V1

## Parent and scope

P25 starts from P24 terminal `bcde7349ad7ba8e6f863959da4239010e6c870be`
and tests exactly one frozen study, SCEPA-Rank. It reuses the frozen Phase2B
detector, signed direction, harm-risk machinery, VisA 12-class inventory,
canonical GT-free caches, exact float32 deployed-score pAP/pAUROC semantics,
and `alpha=0.25`. P25 is not a final architecture and does not authorize P26.

No MVTec/Medical access, CLIP forward, Phase2B training, prompt/adapter
training, alpha/budget/q sweep, image-level value model, or scientific attempt
outside the single P25 marker is permitted.

## Questions and routing

Q1 asks whether a GT-free low-capacity patch ranker transfers a ranking-aligned
benefit target across 12 strict LOCO held classes. Q2 runs only if every Q1
gate passes and asks whether a source-calibrated benefit/risk selector improves
held exact pixel pAP while preserving harm controls.

- Q1 failure: `P25_PATCH_BENEFIT_NOT_IDENTIFIABLE`; Q2 is not run.
- Q1 pass and any Q2 failure: `P25_PATCH_BENEFIT_NOT_POLICY_TRANSFERABLE`.
- Q1/Q2 full pass: `P25_PATCH_ACTIONABILITY_IDENTIFIED` only, which permits
  later explicit review for P26; it is not external validation.

There is exactly one marker and one controller run. Any post-marker failure is
`P25_ENGINEERING_STOP`; no repair or rerun is authorized.
