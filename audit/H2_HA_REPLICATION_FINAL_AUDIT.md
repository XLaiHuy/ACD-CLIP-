# H/A confirmatory replication final audit

Status: `FAIL_HARD_TRAINING_VALIDITY`.

The frozen protocol was followed as H/A-only, E15-only replication for seeds
1 and 2, with a fresh shared E1 per seed, the unchanged Anchor lambda
`0.0021633926715180626`, and family budget `rho=0.10`. Seed 0 remains the
discovery run and is not part of the hard replication gate.

## Training validity

Both confirmatory runs completed with the full shared-E1, H-E2-E15, and
A-E2-E15 checkpoint inventories. All final checkpoints contain finite
full-state tensors. Both runs have zero nonfinite-loss skips, but each has
recorded nonfinite-gradient skips and the resulting H/A optimizer-step totals
are unequal:

| seed | H grad skips | A grad skips | H final step | A final step | validity |
|---:|---:|---:|---:|---:|---|
| 1 | 3 | 2 | 5410 | 5411 | FAIL |
| 2 | 3 | 2 | 5410 | 5411 | FAIL |

The skips are recorded in the training logs and there is no hidden or
repeated-epoch evidence. The frozen rule nevertheless requires equal
successful H/A optimizer-step totals. Therefore neither confirmatory seed is
eligible for target evaluation.

## Target-evaluation firewall

No seed-1 or seed-2 Medical/MVTec evaluation was run. No target checkpoint
selection, target tuning, or result-based restart occurred. No target-ready
freeze manifest was issued because the hard training-validity gate did not
pass. The status-only CSVs in `results/` explicitly record non-evaluation;
they contain no fabricated target metrics.

## Scientific decision

`ANCHOR_REPLICATION_SUPPORT=NOT_CONFIRMED`. A metric-based confirmatory winner
cannot be estimated because both confirmatory trajectories failed the
pre-target validity gate. This run does not authorize a redesign or a new
architecture.

## Provenance

- Seed 1 validity: `audit/H2_HA_SEED1_E15_TRAINING_VALIDITY.json`
- Seed 2 validity: `audit/H2_HA_SEED2_E15_TRAINING_VALIDITY.json`
- Seed 1 root: `runs/h2_ha_replication_e15_seed1`
- Seed 2 root: `runs/h2_ha_replication_e15_seed2`
- Medical evaluator commit: `6bd932fbce0a425af5c8d3f7230dd7dc041568bd`
- Original scientific code SHA: `31167af5ee3dfff80b74af1e9ee0da4ecc475d2e`
