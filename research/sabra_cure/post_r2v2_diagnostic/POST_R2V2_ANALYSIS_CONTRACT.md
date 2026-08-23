# POST-R2V2 Analysis Contract

Parent terminal: `f097be019de365a9598551b4c3c97e33e3d39583`.
R2-v2 preregistration: `b4c67ff15fb2541cbc820b5301d57ae5095aa643`.
R2-v2 execution base: `0a69b6826d132718081fbd7a2edfd25a1b2214c8`.

The execution validates 12 unique held classes, fold/image/patch ordering,
persisted action reconstruction, mutual consistency of accepted/rejected and
correct/wrong masks, finite arrays, `alpha=.25`, historical hashes, and zero
unauthorized accesses/training.  It recomputes D1 and requires strict parity
with published R2-v2 downstream pAP/pAUROC/loss before interpreting D2--D4.

Class tables include native/harm-aware pAP and pAUROC, deltas, coverage,
accepted wrong-sign rate, weighted-harm reduction, BOOST/SUPPRESS/KEEP shares,
harm-risk and `abs(mu)` summaries, sign-correct accepted share, oracle pAPs,
and the frozen loss delta.  Five non-regressing versus seven regressing classes
are compared descriptively only.

Ranking analyses are exact aggregate analyses: per-image AP deltas, rank
displacement for anomaly/normal pixels, positive-vs-negative ordering change,
top-10%-score anomaly enrichment, and score separation.  No result assigns AP
credit to individual patches.

The terminal audit independently recomputes all persisted summary quantities,
checks source immutability and absence of historical writes, verifies all 12
classes once, and confirms `MVTec=0`, `Medical=0`, `CLIP=0`, and
`Phase2B_steps=0`.  Any mismatch is `DIAGNOSTIC_ENGINEERING_STOP` and no
root-cause claim is made.
