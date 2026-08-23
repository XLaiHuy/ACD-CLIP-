# R2-v2 Final Decision

Decision: `R2V2_SCIENTIFIC_STOP`.

This is the sole authorized 12-fold VisA LOCO R2-v2 execution from
`0a69b6826d132718081fbd7a2edfd25a1b2214c8`. All pre/post audits passed.

Safety and harm gates passed: 17.34% coverage, 1.16% accepted wrong-sign,
and 87.54% relative weighted-harm reduction. The downstream pAP gates failed:
harm-aware macro pAP was 0.569825 versus 0.569940 native (−0.0115 pp), and
only 5/12 classes were non-regressing. It did outperform the published failed
R2 selector (0.565567 pAP), but that does not relax the native/breadth gates.

The matched binary control reached 0.568744 macro pAP, below harm-aware; thus
harm weighting is descriptively favored over this control but did not produce
a valid downstream pass. No R3, R4, MVTec, Medical, rerun, threshold change,
or alpha change is authorized.
