# Solution selection and red-team

## Selected solution

`SELECTIVE_PHASE2B_ANCHOR`

The implementation will add one optional, train-only normalized parameter-anchor
term on the CIR image adapter. The reference is the frozen matched Phase2B E14
checkpoint selected before any new target evaluation because it is the
diagnostic parent checkpoint used in the E14 module intervention. The anchor is
not applied to text, soft prompt, RMT state, or inference.

The selected contract is deliberately narrower than a feature-path rewrite:
it directly tests whether constraining the learned image-side movement reduces
the measured R4 signal while leaving the rest of the protocol unchanged.

## Red-team checks

1. **Could the E14 swap be a source artifact?** Yes, the intervention is source
   only and does not establish Medical causality. The same-image feature drift
   is nevertheless observed across E10-E20, and the heldout assessment is
   retained as a negative/mixed control.
2. **Could the anchor merely copy the parent and suppress CIR?** Yes. The
   bounded implementation gate must inspect anchor loss, image update norm,
   source seen/heldout metrics, and whether the solution still has finite
   gradients. A source improvement alone is insufficient for a future target
   claim.
3. **Could K7 be the real cause?** Yes, especially at C E18. Because raw to
   deployed degradation is also present in P and is not consistently C-specific,
   deployment-consistent training is deferred rather than combined with the
   anchor.
4. **Could RMT be intrinsically weak?** The current alpha=.5 signal is neutral.
   This argues against adding a transport redesign now; it does not authorize
   abandoning the mechanism before the corrected image-path experiment.

## Falsifiers and stop conditions

- If the implementation changes optimizer group count, Adam settings, StepLR
  state/timing, precision, batch geometry, prompt/DFG schedule, evaluator, or
  deployment operator, stop and repair the implementation.
- If the anchor leaves the image-path drift unchanged in the bounded gate, or
  causes non-finite loss/gradient, the candidate is rejected.
- If the bounded source gate cannot compare the selected solution at a matched
  training horizon, it is marked INCONCLUSIVE and no full run is authorized.
- No Medical/MVTec result may be used to tune the anchor coefficient in this
  stage.
