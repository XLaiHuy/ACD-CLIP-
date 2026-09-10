# Dataset Transfer Manifest

Raw datasets are present on the source machine but are intentionally not
committed to GitHub. The CSV beside this file records the exact source paths,
file counts, byte totals, class names, metadata hashes, and transfer role.

The expected laboratory paths in this handoff are examples. Change the
`/lab/data/acd-clip` prefix if the laboratory storage layout differs, then
create the repository `data/` symlinks described in `MACHINE_HANDOFF_LAB.md`.

## Source inventory

| Dataset | Source files | Source size | Manifest | Role |
|---|---:|---:|---|---|
| VisA | 12,037 | 1,920,559,633 bytes | `dataset/hub/VisA.jsonl` | Run V source and existing frozen provenance |
| MVTec-AD all | 6,644 | 5,271,247,124 bytes | `dataset/hub/MVTec_all_supervised.jsonl` | Run M source |
| MedAD | 27,007 | 2,601,429,819 bytes | Brain, Liver, Retina JSONL files | Medical-6 target |
| Colon | 3,984 | 583,061,019 bytes | Colon JSONL files | Medical-6 target |

The MVTec source pool used by Run M is the new 5,354-row manifest: 3,629
`train/good`, 467 `test/good`, and 1,258 labelled `test/<anomaly_type>` images.
All 1,258 anomaly rows have official masks.

## Safe transfer examples

Do not run these from the source machine until the destination is mounted and
the destination paths have been reviewed:

```bash
rsync -aH --numeric-ids --info=progress2 /workspace/datasets/med-visa/data/VisA_20220922/ /lab/data/acd-clip/VisA_20220922/
rsync -aH --numeric-ids --info=progress2 /workspace/datasets/mvtec-ad/ /lab/data/acd-clip/mvtec-ad/
rsync -aH --numeric-ids --info=progress2 /workspace/datasets/med-visa/data/MedAD/ /lab/data/acd-clip/MedAD/
rsync -aH --numeric-ids --info=progress2 /workspace/datasets/med-visa/data/Colon/ /lab/data/acd-clip/Colon/
```

Equivalent `scp` examples, for smaller or manually staged transfers:

```bash
scp -r /workspace/datasets/med-visa/data/VisA_20220922 lab-machine:/lab/data/acd-clip/
scp -r /workspace/datasets/mvtec-ad lab-machine:/lab/data/acd-clip/
scp -r /workspace/datasets/med-visa/data/MedAD lab-machine:/lab/data/acd-clip/
scp -r /workspace/datasets/med-visa/data/Colon lab-machine:/lab/data/acd-clip/
```

After transfer, run `bash handoff/lab_20260910/bootstrap_lab.sh`. The bootstrap
does not copy data, start training, or evaluate any target dataset.
