# Dataset restore notes

Raw datasets are intentionally not committed to GitHub.

## Required data

- Source training dataset: VisA.
- Later target datasets: MVTec, Brain, Liver, Retina,
  Colon_clinicDB, Colon_colonDB, and Colon_Kvasir.

The tracked dataset manifests and setup notes are under `dataset/hub/` in the
main repository. The known VisA manifest identity is:

`468463d2d6234fa7537c6da32b027758527676a12a54a4028c5a282cdd726842`

Restore the datasets using the same directory layout expected by the tracked
manifests and verify the manifest identity before any future training or
target evaluation. Do not use raw dataset files from a different split or
preprocessing contract.

The CLIP model weight is already tracked through Git LFS at
`model/ViT-L-14-336px.pt` with SHA256:

`3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`
