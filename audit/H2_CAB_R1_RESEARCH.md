# H2 CAB-LoRA R1 Research Record

Frozen root-cause evidence says the frozen ViT already contains contextual
information, while Conv-LoRA is the primary trainable context amplifier and
Stage 2 is the strongest observed spillover site. S2-LOCR showed that direct
Stage-2 score suppression can reduce near-background responses while also
reducing anomaly/interior evidence. The Functional Anchor showed that global
feature preservation is non-selective. NFUR-R2 was not supported, so CAB-R1
does not add a downstream refinement head.

CAB-R1 therefore tests one falsifiable hypothesis: penalize only the positive
increase in context sensitivity introduced by Stage-2 Conv-LoRA, relative to
the frozen/pre-adapter representation. Native ViT contextuality is not
matched globally and no score suppression, feature teacher, DFG change, SS2D
change, routing, or multi-stage term is present.

The experiment is source-only on VisA train. It is a bounded 500-attempt
CONTROL versus CAB-LoRA R1 screen and stops at the decision artifact. Medical
and MVTec inference are prohibited before a passing bounded decision and are
not part of this driver.
