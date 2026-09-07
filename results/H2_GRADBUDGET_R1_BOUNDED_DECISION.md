# H2 GradBudget R1 bounded decision

- Protocol: `H2_NEXT_MECHANISM_BOUNDED_R1_LATE_CONVLORA_ABNORMAL_DICE`
- Branch: `research/h2-late-convlora-gradbudget-r1`
- Starting E1 SHA256: `7f9176b7ef53b572935567c574535075a573175b2aa83505d043a71d45b12b35`
- Scope: stage-2 and stage-3 Conv-LoRA only; stage 1 excluded.
- Endpoint: frozen 96-image VisA source split only; no Medical/MVTec or target inference.

## Decision: `INVALID_NUMERICAL`

The numerical gate is invalid because control had 499 successful updates and candidate had 500; the protocol requires equal successful counts. No rerun or parameter change is authorized.

Mechanism activity was established in telemetry: candidate activity fraction 0.580000, alpha median 0.978670, and effective abnormal/rest was lower than raw whenever active.

Candidate-minus-control endpoint deltas:

- final AP: `+0.009913566`; final AUROC: `-0.083032987`
- positive mean/median: `+0.067497611` / `-0.000002279`
- interior mean/median: `+0.079689100` / `+0.000007445`
- near-background p95/p99: `+0.267286159` / `+0.408107847`
- stage-1 AP/AUROC: `+0.065048836` / `-0.064128897`
- stage-2/stage-3 Conv-LoRA drift: `+3.222007312` / `+2.348765069`

Red-team interpretation: drift reduction was not achieved; the endpoint tail and AUROC checks worsen despite an AP increase. This does not support a confirmatory run.

No target inference was run. The bounded screen is complete and the workflow is waiting for explicit user approval before any further experiment.
