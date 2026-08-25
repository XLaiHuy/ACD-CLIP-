# P26 SABRA Final Machine Handoff

The frozen architecture is `SABRA-FINAL-NATIVE-PHASE2B-V1`. Clone the
repository, check out `research/p26-sabra-cure-final-architecture-freeze-v1`,
hydrate Git LFS, then run:

```bash
bash scripts/restore_p26_sabra_final.sh
```

The script verifies the branch/revision, clean tree, canonical config hash,
every required artifact, and the no-science `--check-only` path. It prints
`SABRA_FINAL_RESTORE_STATUS=READY` only when the machine is ready.

No separate transfer bundle is needed: both required checkpoints are tracked
through Git LFS and their local hydrated hashes were verified. If a clone has
only LFS pointers, hydrate the two exact paths named in
`P26_CHECKPOINT_MANIFEST.json`; never substitute another checkpoint.

READY does not authorize evaluation. MVTec remains untouched by P26, Medical
is forbidden, and `--run` remains locked pending explicit user authorization.
