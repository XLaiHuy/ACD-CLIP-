# Phase5-B0 Second-Evidence Discovery Audit

Decision: `SECOND_EVIDENCE_CLASS_UNSTABLE`

Input integrity: `PASS`; inference forwards: `2162`.

This was an inference-only detection audit over held-out VisA TEST. The four candidate families were fixed before evaluation; no selector was learned and no dense feature cache was persisted.

- **E_local**: matched-pair win rate 0.423488; positive C_AP capture 0.012839; status `NOT_SUPPORTED`.
- **E_multistage**: matched-pair win rate 0.622848; positive C_AP capture 0.040506; status `NOT_SUPPORTED`.
- **E_xstage**: matched-pair win rate 0.568330; positive C_AP capture 0.029578; status `NOT_SUPPORTED`.
- **E_global**: matched-pair win rate 0.419675; positive C_AP capture 0.014441; status `NOT_SUPPORTED`.

Primary candidate: `None`.

Next: Resolve class-level instability of the candidate signal before any method design; do not launch external representation search.
