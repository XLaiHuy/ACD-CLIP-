# P20 Final Decision

P20 completed its single authorized attempt with 12/12 science folds, 12/12
independent fold audits, and passing global value-metric and global-audit
workers. The JSON-safe `NO_EXPANSION` sentinel round-tripped under strict JSON;
no child failure occurred.

The frozen P14 result is **`P14_SCIENCE_RECOVERED_STOP`**. This is a valid
scientific stop, not an engineering failure and not a new method result.

Key frozen aggregate evidence: context macro pAP `0.5697706390` versus native
`0.5699400925`; context macro pAUROC `0.9714339111` versus native
`0.9707623684`; coverage `0.1835722139`; wrong-sign rate `0.0121380443`;
relative weighted-harm reduction `0.9901982208`; global value Pearson
`0.0339006212`; global stable-rank Spearman `0.1570901118`.

Passing gates: G1, G2, G3, G9, G11. Failing gates: G4, G5, G6, G7, G8, G10.
No P14 gate is relaxed or reinterpreted. MVTec and Medical access remain zero;
additional CLIP forwards and Phase2B optimizer steps remain zero.

Next allowed action: explicit user review. At most one final post-P14
diagnostic may be considered under a new explicit authorization; no new model
or rerun is authorized automatically.
