# Next-machine start

Read `PROJECT_HANDOFF_PHASE5_20260818.md` first. Do not immediately rerun
P5FR1C/P5FR1CE1, rerun the 1,725-record derive, reopen medical, or begin P5-G.

```bash
git clone https://github.com/XLaiHuy/ACD-CLIP-.git
cd ACD-CLIP-
git fetch --all --prune
git lfs install
git lfs pull
git checkout handoff/phase5-20260818-portable
git rev-parse HEAD
```

Expected authoritative forensic parent:
`3f41c7e6ee4db86c67781edd3c71c80fbc1daa72`. The handoff HEAD will be this
commit plus the handoff commits.

## Checkpoint recovery

The required checkpoint is already remote in Git LFS. Do not expect a copied
`.pth` blob in the handoff branch:

```bash
git fetch origin artifacts/p5-runtime-inputs
git lfs pull origin artifacts/p5-runtime-inputs --include='runs/phase4v/v1_7/readiness_full/adapter_5.pth'
sha256sum runs/phase4v/v1_7/readiness_full/adapter_5.pth
```

Expected checkpoint SHA256:
`a786e1498254d4589824e7bd111e1917b399d31687ed807212e86dfe3893df34`.
Expected size: `56451915` bytes. Expected config SHA256:
`377ce1c0ae1dd870f82ddcb828d8d8809fa46c007e61567f2150ec11354b23a4`.

## Environment

Recreate `torchhuy` from `handoff/environment/RECREATE_ENV.md`, then record
fresh conda and pip exports. The source machine did not expose conda or the
preferred environment, so those files explicitly record that limitation.

## Dataset

MVTec images and masks are not committed. Restore/download the archive only
for a future authorized industrial study, then verify the archive,
`dataset/hub/MVTec.jsonl`, and canonical identity hashes in
`handoff/EXTERNAL_ASSET_MANIFEST.json`. Do not run science during restoration.

## Runtime caches

Optional large evidence caches remain only on the rental filesystem. Consult
`handoff/MANUAL_TRANSFER_REQUIRED.md` before shutting it down if byte-level
cache preservation is desired. Current scalar/result conclusions do not
depend on transferring those caches.
