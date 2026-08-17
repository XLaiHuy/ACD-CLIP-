# Phase5 worktree inventory

Inventory performed read-only on 2026-08-18 before creating the handoff
branch. Existing worktrees were not reset, cleaned, stashed, staged, or
modified. The handoff worktree was created afterward from the authoritative
forensic commit.

| Path | Branch | HEAD | State | Relevant dirty content |
|---|---|---|---|---|
| `/workspace/ACD-CLIP-` | `autopilot/p5-minimal-reference-adjudication` | `1d636895e9f6f299d14f92ae150aebf4e3b4fb80` | dirty | 4 modified and 4 untracked E0R HRIP recovery artifacts; intentionally preserved and snapshotted |
| `/tmp/acd-p5-runtime.ulVPT8` | detached | `316ce9d4a9ddf742cf17f1c98c5011891c90ab08` | clean | none |
| `/workspace/ACD-CLIP-P5F` | `autopilot/p5-f-mvtec-four-family-v2` | `859d4e2c8074f05ffb2056e4efe5fc38657763bb` | clean | none |
| `/workspace/ACD-CLIP-P5FR1` | `autopilot/p5-fr1-mvtec-four-family-v2` | `529096949afae2c6bad2b73d976a6f8cf0455e3d` | clean | none |
| `/workspace/ACD-CLIP-P5FR1C` | `autopilot/p5-fr1c-mvtec-late-reconciliation` | `cd0d22072d34804494e38fa95bc2e9d338bead7c` | dirty | 2 untracked evaluator failure-status artifacts; intentionally preserved and snapshotted |
| `/workspace/ACD-CLIP-P5FR1CE1A` | `autopilot/p5-fr1ce1-a-final-forensic-reconciliation` | `3f41c7e6ee4db86c67781edd3c71c80fbc1daa72` | clean | none |
| `/workspace/P5FR1CE1` | `autopilot/p5-fr1c-e1-evaluator-recovery` | `2ef784ff91b91e3b2c2c880dfaa74c02e94445d2` | clean | none |
| `/workspace/ACD-CLIP-PHASE5-HANDOFF` | `handoff/phase5-20260818-portable` | `3f41c7e6ee4db86c67781edd3c71c80fbc1daa72` at creation | clean before handoff files | newly created handoff worktree |

## Snapshotted relevant dirty/untracked files

All ten files were scanned for common secret patterns and copied byte-for-byte.
The exact dirty bytes were not present in the authoritative forensic commit;
modified tracked files have only their older committed versions on the source
branch. No caches, raw datasets, images, masks, or build artifacts were copied.

Source `/workspace/ACD-CLIP-`, source HEAD
`1d636895e9f6f299d14f92ae150aebf4e3b4fb80`:

| Status | Source relative path | Handoff snapshot | SHA256 |
|---|---|---|---|
| M | `runs/phase5/hsir/P5E0R_HRIP_EVALUATION_RECOVERY/DECISION.json` | snapshot subtree | `ba9d36c051e7aaa18518150b5df8550126094787a8dd0703753f15509ce521f9` |
| M | `runs/phase5/hsir/P5E0R_HRIP_EVALUATION_RECOVERY/OUTPUT_CHECK.json` | snapshot subtree | `7411fa587d6ec7a8ca8e8dc68b24ed2968c4fb05420628be711aa6dbfee74f31` |
| M | `runs/phase5/hsir/P5E0R_HRIP_EVALUATION_RECOVERY/RECOVERY_PROVENANCE.json` | snapshot subtree | `15de08d1ffebba1a1dbf118688634e495a66604bb3045193fd30885fda7f4310` |
| M | `runs/phase5/hsir/P5E0R_HRIP_EVALUATION_RECOVERY/REPORT.md` | snapshot subtree | `abc861a0965b2b468e65ef66fc3b79676be628b9177393751eba84bd99b0b299` |
| ? | `runs/phase5/hsir/P5E0R_HRIP_EVALUATION_RECOVERY/ALIGNED_SHIFTED.json` | snapshot subtree | `05252c807b6a0da60d2183a1450d862b729a54527ab0666bd75e89cd04506d18` |
| ? | `runs/phase5/hsir/P5E0R_HRIP_EVALUATION_RECOVERY/LEVERAGE_SAFETY.json` | snapshot subtree | `06b8e85d7635ba337ad80c0474fb25878bcf5b725a256a8bcfcc69825420a3b4` |
| ? | `runs/phase5/hsir/P5E0R_HRIP_EVALUATION_RECOVERY/PER_CLASS.csv` | snapshot subtree | `92e33075093aa993e56603162d186c850fe33e2bcf72f42bd8778ed6259ce701` |
| ? | `runs/phase5/hsir/P5E0R_HRIP_EVALUATION_RECOVERY/PRIMARY_SIGNAL_AUDIT.json` | snapshot subtree | `92dfee4d99aafb500e66201831663a91d51837d83470c040827af3b69176835b` |

Source `/workspace/ACD-CLIP-P5FR1C`, source HEAD
`cd0d22072d34804494e38fa95bc2e9d338bead7c`:

| Status | Source relative path | Handoff snapshot | SHA256 |
|---|---|---|---|
| ? | `runs/phase5/hsir/P5FR1C_MVTEC_LATE_COMPLETION/EVALUATION_COMMAND_STATUS.json` | snapshot subtree | `693d43f7a4e13fdff51bf42544848f3481bb951afbb95d3f4f97588df5c8dc8b` |
| ? | `runs/phase5/hsir/P5FR1C_MVTEC_LATE_COMPLETION/EVALUATOR_RUN.json` | snapshot subtree | `5a78fe9c7f6ce772bb6e804a7cad7e90b1738930cc5e0e2a3f2b612a6ddbe342` |

These snapshots are context/provenance only. They do not alter the completed
forensic result and must not be used to rerun P5FR1C.
