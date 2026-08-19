# Next-machine start

Start from the synchronized final handoff commit at the branch tip. The persisted Recovery-v2 candidate is `M1_E_Credibility`; preserve it exactly. The readiness boundary before this handoff is `a1de69cf0d365ab34e1ac6257453806e7003a22d`.

`FULL_20E_TRAIN_AUTHORIZED=false` and `EXPLORATORY_20E=false`. Do not start 20e training, access medical data, or represent an exploratory run as authorized. MVTec external validation remains unavailable because the required image root was absent; no external metric is claimed. If an authorized MVTec image root is later supplied, only the already-frozen external evaluation may be considered, with zero tuning and no candidate redesign.

Do not rerun VisA, Phase2B, or rebuild any cache. Verify Git LFS objects and the final package checksums before any future work.
