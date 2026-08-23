# P19 — High-Performance Isolated Global-Metric P14 Recovery V1

P19 is an engineering-only recovery from P18 terminal
`a2c4b704e152921e7fbf498f9d6ceb71b6769e2d`. P18 created no attempt. P19
preserves the P14 source hash `d83c59c7e52b21022c90708198048f049bd4e4e46ff443a67adf0c524414273f`,
the accepted P15 optimized AP engine, all P14 folds, targets, features,
lambda, q set, alpha, metrics, aggregation semantics, gates, and firewalls.

The sole architectural change is ownership: each outer fold and each fold
audit runs in a short-lived child; a separate global-value child owns only
image-level `(image_index,vhat,V_j)` pair files. The parent reads JSON and
hashes only. P19 has one possible attempt, containing 12 science workers,
12 audit workers, one global metric worker, and one global metric audit.

Pre-prereg feasibility is GO: P18 proved every non-value-ranking aggregate is
compactly sufficient; P14's sole non-compact component is global stable-rank
Spearman, which is exactly reproduced in the isolated global-value role.
No P15/P16/P17 partial artifact is a P19 input or result. MVTec, Medical,
new CLIP forwards, and Phase2B optimization are forbidden.

The execution-only `INNER_AP_WORKERS` choice is selected deterministically
from `{1,2,4}` on engineering fixtures after implementation, using exact
parity, <=12 GiB benchmark RSS, and >=10% median speed gain over one worker;
otherwise it is `1`. The selected worker and CPU thread configuration are
frozen in the published execution base before `ATTEMPT_STARTED`.
