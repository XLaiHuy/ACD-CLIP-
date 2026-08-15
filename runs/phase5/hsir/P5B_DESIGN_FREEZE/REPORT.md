# P5B overnight design freeze report

## Outcome

`selected_candidate = NONE` and `terminal = P5B_RELIABILITY_AUDIT_REQUIRED`.

The evidence supports the need for a bounded intervention family but does not support freezing a deployable candidate. No candidate implementation, training, new VisA candidate evaluation, medical evaluation, or Phase2B/model predictor modification was performed.

## Input and firewall

The branch/HEAD preflight passed at the expected P5B starting point. Existing B2, B3, B3.1, actionability, and reference-validity artifacts were audited. Source hashes and artifact checksums are in `INPUT_CHECK.json`. The GT firewall passes: predictor and C1 consume GT-free quantities, while B3 bridge pairs are produced only after labels are loaded for post-hoc evaluation. This review performed zero model forwards and zero training steps; the only runtime probe used synthetic tensors through the exact deployment function.

## Mechanism evidence

B2 established aligned E pairwise support (`W=0.676282...`, 11/12 supportive classes), but the 2,785 B3/B3.1 rows are not an inference population. Aligned B3.1 utility is broad in base rank distance: gap >10 contains 461 of 592 rescues and net +308; gap 1 contains 18 rescues and net +14. This rejects adjacent-only design. The gap >10 class-bootstrap net CI is `[5.0833,49.9167]` across 12 classes, while gaps 2 and 3 are unstable.

Rank-gap/spatial-distance correlations are near zero, and rescued/broken spatial distances overlap. Thus rank-locality is not spatial-locality. B3 displacement harm and native-to-deployed reversal support bounded deployment-aware authority, but do not specify a selector.

## Gate 3 decision

A future candidate needs a GT-free pair proposal, relation acceptance, abstention, conflict/repetition handling, unrelated-order preservation, and bounded spatial authority. None is present in committed artifacts. B3/B3.1 evaluation bridges cannot be promoted into inference constraints. C1’s full E sort cannot be a fallback because it is the closed broad-reranking behavior.

Candidate A is rejected by rank geometry. Candidate B is the mechanistically preferred family but remains unfreezable. Candidate G is rejected for C1-like cascading risk. The correct decision is `NONE`, not a forced candidate.

## Deployment and transfer

The exact pre-softmax path is a positive linear smoothing/interpolation operator, followed by stage mean and softmax. The corrected synthetic probe passed: positive/negative native impulses preserve sign before softmax, but signed corrections create both positive and negative spatial effects. This proves an operator property only; it does not prove deployed ranking improvement. Medical transfer remains unproven; only relative same-image evidence and abstention are plausible transfer primitives, while VisA constants and industrial rules are not transferable by assumption.

## Smallest next audit

Run the temporary GT-free selector reliability audit specified in `PREREGISTRATION.md`. It must persist pre-action per-patch quantities and a declared proposal trace before any GT join. If arrays are not recoverable, one inference-only 2162-image instrumentation pass is the smallest forward requirement. Do not start a full candidate evaluation until that audit passes Gate 3.
