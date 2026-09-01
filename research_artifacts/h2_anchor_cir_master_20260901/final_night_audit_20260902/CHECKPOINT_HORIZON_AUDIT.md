# Checkpoint horizon audit

The historical H2 run was not an E1-E20 run. Its log args specify `epoch=15`, the log reaches epoch 15, and the directory contains `adapter_1.pth` through `adapter_15.pth` only.

| Run | Available/verified horizon | Candidate states | Comparison status |
|---|---|---|---|
| Historical H2 | E1-E15 | E1-E15 model-only checkpoints | Historical parent oracle |
| Current R | E1-E10 training history; E10 candidate checkpoint | E10 | Same-E10 control |
| Current RA | E1-E20 training history; E10/E12/E14/E16/E18/E20 candidates | E10, E12, E14, E16, E18, E20 | Extension; E16/E18/E20 outside historical parent horizon |
| Current RCA | E1-E20 training history; E10/E12/E14/E16/E18/E20 candidates | E10, E12, E14, E16, E18, E20 | Extension; E16/E18/E20 outside historical parent horizon |

Therefore any comparison involving RA/RCA E16, E18, or E20 is not a matched historical-H2 horizon comparison. The same-E10 Medical comparison in `SAME_E10_MEDICAL.md` avoids that specific horizon confound, but it still inherits the current Anchor and trajectory provenance limitations.
