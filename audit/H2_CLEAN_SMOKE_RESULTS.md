# H2 clean bounded smoke

Command:

`RUN_SMOKE=YES SMOKE_ROOT=/tmp/h2_clean_smoke_v1 bash scripts/run_h2_clean_smoke.sh`

The shared native E1 run processed five batches and produced a version-2
full-state checkpoint. H, A, C, and AC each resumed from that same E1 state,
processed five E2 batches, and produced `adapter_2.pth`.

| arm | anchor | anchor gradient ratio | CIR | CIR peer valid | CIR delta detached | result |
|---|---:|---:|---:|---:|---:|---|
| H | no | n/a | no | n/a | n/a | PASS |
| A | yes | `3.60312478733249e-06` | no | n/a | n/a | PASS |
| C | no | n/a | yes | `1.0` | `True` | PASS |
| AC | yes | `2.6562543837371775e-06` | yes | `1.0` | `True` | PASS |

All five checkpoints contain the required model, optimizer, scheduler,
GradScaler, RNG, DataLoader-generator, config/hash, provenance, and
parameter-only anchor-reference fields. No non-finite loss or parameter was
reported. The smoke is a wiring/finite-gradient gate only; it is not a
performance comparison and does not authorize geometry or full training.
