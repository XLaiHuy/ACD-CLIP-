# Machine handoff: H/A E15 confirmatory replication

## Decision

`ANCHOR_REPLICATION_SUPPORT=NOT_CONFIRMED`.

Both confirmatory seeds failed hard training validity before target
evaluation. Do not interpret the status-only result CSVs as target metrics.

## Persistent checkpoints

| seed | role | path | SHA256 |
|---:|---|---|---|
| 0 | discovery shared E1 | `frozen_seed0_final/shared_e1/adapter_1.pth` | `7f9176b7ef53b572935567c574535075a573175b2aa83505d043a71d45b12b35` |
| 0 | discovery H15 | `frozen_seed0_final/H/adapter_15.pth` | `6830137b52fc16321192909fe7b3ca7565664afe61a8592fe7b7a78d2dff4771` |
| 0 | discovery A15 | `frozen_seed0_final/A/adapter_15.pth` | `727dc4813db1ef0c5a6db3a4cf15e916ad1413dcf629913358419d2734edecfe` |
| 1 | confirmatory shared E1 | `runs/h2_ha_replication_e15_seed1/shared_e1/adapter_1.pth` | `c893693d42a1d5221b4980a62e762de2dcf150e4b3871fbe3f0ff943d1d29888` |
| 1 | confirmatory H15 | `runs/h2_ha_replication_e15_seed1/H/adapter_15.pth` | `2547b621341fe0ab79f516b406190b8ca35afab84e18427a525ec30f9cedd260` |
| 1 | confirmatory A15 | `runs/h2_ha_replication_e15_seed1/A/adapter_15.pth` | `40db05af5ff9d75e6cae792a2df0cbdcc1afc1181cc88a9dac18ef2df6aaba99` |
| 2 | confirmatory shared E1 | `runs/h2_ha_replication_e15_seed2/shared_e1/adapter_1.pth` | `9da4ec04ac72c2a9937557e049a5c6ca1f9deb7369ce29aff5bd98da7419fc3d` |
| 2 | confirmatory H15 | `runs/h2_ha_replication_e15_seed2/H/adapter_15.pth` | `c2a3725edd92dd65fc70e7d3b89579a913400aa0c122b9447378cc45dc2597fe` |
| 2 | confirmatory A15 | `runs/h2_ha_replication_e15_seed2/A/adapter_15.pth` | `caba459d532b1bca0494b703b6f179fca0f40a163eed9c5da3c65a8dd8d1fda5` |

## Shared identity

- Original scientific code: `31167af5ee3dfff80b74af1e9ee0da4ecc475d2e`
- Replication implementation: `67888aa3eba2e7d2eecf90bfb8ba132c2c137aa`
- Base H2 commit: `e03966997d4cecfd985943a4053a93e1e40197ec`
- CLIP SHA256: `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`
- VisA manifest SHA256: `468463d2d6234fa7537c6da32b027758527676a12a54a4028c5a282cdd726842`
- Medical evaluator commit: `6bd932fbce0a425af5c8d3f7230dd7dc041568bd`
- Anchor lambda: `0.0021633926715180626`; family budget rho: `0.10`

## Validity and firewall

- Seed 1: H/A steps `5410/5411`; nonfinite-loss skips `0/0`; grad skips `3/2`; target evaluation not run.
- Seed 2: H/A steps `5410/5411`; nonfinite-loss skips `0/0`; grad skips `3/2`; target evaluation not run.
- Checkpoint roots are persistent and intentionally untracked; do not commit checkpoints.
