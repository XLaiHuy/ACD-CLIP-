# External asset manifest

## MVTec AD

- Dataset: MVTec AD
- Source: Kaggle `ipythonx/mvtec-ad`
- Current data root: `/workspace/data/mvtec_ad`
- Original archive: `/workspace/data/_downloads/p5f_mvtec_20260817T062919Z/mvtec-ad.zip`
- Archive SHA256: `61124d44b1e62ad0dc64e1b6111c7ffcfda20cd36a92f68e14df0a8016cf477b`
- Metadata: `dataset/hub/MVTec.jsonl`
- Metadata SHA256: `3a5e304ea16bba82e6e525d188698e91ca92b718696f8c257ed435d235b4cc2c`
- Canonical identity SHA256: `c0ace7f629a636db6393aca7bebe1b37a6a9f5673ff59ff8b6800484642faa34`
- Records: 1,725
- Classes: 15
- MVTec images/masks committed: false

## Reconstruction on a new machine

Do not commit the extracted dataset. Obtain the archive from the recorded
Kaggle source or an authorized copy, verify the archive SHA256, extract it to
`/workspace/data/mvtec_ad`, and verify that `dataset/hub/MVTec.jsonl` and the
canonical identity hash match the committed protocol. Do not run evaluation
as part of reconstruction. The archive is not required to recover the current
scientific conclusion; it is required only for a future, freshly authorized
industrial study.
