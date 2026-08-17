# E0R Evaluator Semantic Diff Review

The historical evaluator is
`tools/audit_phase5_p5e0_hrip_posthoc.py`. It reconstructs frozen unshifted
arrays, reads GT only in its post-hoc phase, and uses the authoritative
matching, triage, and leverage helpers. Its confirmed defect is that it applies
`shifted_map` to a class-concatenated HRIP array rather than to one image map.

The new evaluator is
`tools/audit_phase5_p5e0r_hrip.py`. It is a separate zero-forward evaluator
and does not modify or import the historical E0 evaluator. Its semantic diff is
limited to:

1. a new E0R output namespace and recovery provenance;
2. explicit frozen-cache/hash validation;
3. explicit zero-forward enforcement;
4. per-image application of the existing authoritative 518x518
   `shifted_map` before class concatenation.

All unshifted semantics remain frozen: native-logit score reconstruction,
D_rank reconstruction, labels and pixel IDs, matching, top-20% risk
population, top-10% triage, C_AP/R_pos/R_neg, bootstrap repetitions, seeds,
gates, candidate, and decision precedence. No HRIP, E_nonlocal, peer, tau, or
alpha computation is present in E0R.

Any difference beyond this list is a scope-expansion failure.
