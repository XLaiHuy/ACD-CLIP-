# P24 Next-Study Justification

The only research question justified by the audited evidence is:

> Can a frozen-detector, GT-free, low-capacity patch-level selector identify
> sign-correct proposals with positive ranking-level intervention value while
> preserving the already-established harm controls?

Any future preregistration must keep three distinct concepts separate:

1. **Direction** — whether BOOST/SUPPRESS has the correct sign.
2. **Harm** — risk of the correction.
3. **Benefit/actionability** — whether a safe action improves ranking/pAP.

Proposed future architecture, not an established result:

```
Frozen Phase2B detector
  -> frozen signed proposal direction
  -> harm-risk estimator
  -> patch-level benefit/actionability selector
  -> BOOST / SUPPRESS / KEEP
  -> fixed alpha correction
```

The justification is P13's GT-bearing patch-level diagnostic opportunity,
P23's closure of the coarse image-level action family, and retained safety
machinery.  The critical unknown is a leakage-safe, GT-free benefit target
with cross-class ranking alignment.  No P25 or other experiment is authorized
by this document.
