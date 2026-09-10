# Source Machine Inventory

## Git source state

- Repository root detected with `git rev-parse --show-toplevel`: `/workspace/ACD-CLIP-`.
- Frozen source HEAD: `aa77c76baa49cd2fb621fde7a11a8850313782de`.
- The complete pre-modification Git/LFS preflight output was saved at
  `/tmp/acd_clip_handoff_preflight_20260910.txt` on the source machine.
- Handoff branch was created from that frozen commit.
- The tracked source, model, configuration, audit, result, forensic archive,
  dataset hub, script, and test trees are already part of the frozen commit.
- The required R3 final-freeze reports are preserved byte-for-byte from the
  frozen commit. This branch adds migration files only, apart from the
  MVTec-all loader alias in `dataset/info.py`.

## Local-only classification

| Classification | Local scope | Migration decision |
|---|---|---|
| REQUIRED_FOR_REPRODUCTION | Tracked source/config/result/audit trees, CLIP base weight, required LFS checkpoints, existing dataset JSONL files, and the new MVTec-all manifest | Keep in Git; LFS already tracks the large checkpoint and CLIP paths |
| USEFUL_AUDIT_ARTIFACT | Tracked `forensic_archive/`, R3 logs/reports, and the source provenance files | Keep in Git as already frozen; no ephemeral rerun needed |
| LARGE_BINARY | Tracked CLIP and checkpoint files listed in `CHECKPOINT_MANIFEST.csv` | Keep as existing Git LFS objects; no duplicate checkpoint copies |
| RAW_DATASET | Ignored `data/` symlinks and source data under `/workspace/datasets/` | Do not commit; transfer separately using the dataset manifest |
| SECRET_OR_PRIVATE | No candidate secret filenames or credential patterns were found in the repository or root shell history | No secret file is copied or staged; values were not printed |
| TEMPORARY_OR_DISPOSABLE | Ignored Python caches, ephemeral `/tmp/r3_*` logs/checkpoints, and the empty download staging area | Do not commit; final tracked forensic artifacts remain the source of truth |

## Ignored files

Before handoff additions, `git status --ignored --short` showed only Python
cache directories/files and the four ignored `data/` symlinks. The symlinks
resolve to VisA, MedAD, Colon, and MVTec raw datasets. No untracked source or
result file was present before the handoff branch was created.

## R3 command provenance

The exact completed R3 scripts and their audit/config/result artifacts remain
under the tracked `scripts/`, `configs/`, `audit/`, `results/`, and
`forensic_archive/` trees. The frozen R3 E15 checkpoint is listed separately
in `CHECKPOINT_MANIFEST.csv`. Ephemeral `/tmp/r3_*` material was classified as
temporary or useful audit material only; it is not needed because the final
forensic archive and final-freeze reports are tracked.
