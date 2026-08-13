# Phase4 Stage 1.7 terminal decision

Terminal: `PHASE4_TEXT_SEMANTIC_LINE_STOPPED`.

The deterministic Train-only manifest contains 48 images: two Normal and two
anomaly-labelled images from each of the 12 VisA classes.  All 24 anomaly
images retain nonzero pooled anomaly-mask support.  Selection is a published
SHA256 rank over `(seed, class, label, image path)`, not model outputs.

K1 A1 has conditional anomaly-region utility (mean `+0.48435` at patch level;
`68.44%` of anomaly patches favor A1), while every Normal patch favors A0
(mean utility `-0.16545`).  This is not a lack-of-capacity result.

No allowed selector signal meets the image-level safety requirement:

- Raw affinity has all-patch AUROC `0.52334` for `A1_better`.
- Tangent residual is worse: all-patch AUROC `0.46959`.
- Raw state/class magnitude has image-level utility correlations `0.04384` /
  `0.04454`.
- Parameter-free contrastive local evidence has all-patch AUROC `0.73616`,
  but this is a prevalence artifact: its anomaly-only AUROC is `0.49836`.
- Direct residual gating with `sigmoid(score(A1)-score(A0))` keeps anomaly
  gain `+0.01654` but harms Normal patches `-0.00151`; the semantic-vector
  NO-OP mixture is slightly safer on Normal patches (`-0.00129`) and also
  fails the safety gate.

The base local posterior is not a supported local selector either: anomaly
patch AUROC is `0.51641`, with mean posterior `0.02624` on anomaly patches
versus `0.02572` on Normal patches.  This audit has neither foreground/
background nor texture annotations, so it does not diagnose either mechanism.

No Stage 1.8 selector, K2, OT, Router, ACT, E20, or medical evaluation is
authorized from these results.  The single future pivot recommendation is an
inference-only **visual-only adaptation audit**: the current text-conditioned
direction is conditionally useful but cannot be safely selected, while the
current base local posterior is near chance.

Authoritative machine-readable artifacts:

- `visa_train_audit_manifest.json`
- `K1_ORACLE_UTILITY_AUDIT.json`
- `K1_DIRECT_RESIDUAL_GATE_AUDIT.json`
- `K1_VISUAL_TEXT_PIVOT_AUDIT.json`
