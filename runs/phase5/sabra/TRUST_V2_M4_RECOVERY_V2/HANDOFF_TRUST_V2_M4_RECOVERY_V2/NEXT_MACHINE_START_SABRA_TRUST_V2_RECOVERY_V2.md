# Next-machine start

Start from the synchronized failure-state handoff commit at the branch tip. The persisted Recovery-v2 candidate is `M1_E_Credibility`; preserve it exactly. The terminal MVTec stage is `EXTERNAL_VALIDATION_FAILURE` with `VALID=false` because the required image root was absent. No MVTec image or mask was read and no external metric is claimed.

The VisA result and finalized GT-free cache remain valid. If an authorized MVTec image root is later supplied, verify the candidate freeze and run only the frozen external evaluation. Do not rerun VisA, rebuild any cache, retune, redesign, access medical data, or authorize full-20e training from this state.
