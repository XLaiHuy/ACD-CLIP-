# H2 master Medical evaluation

Status: COMPLETE. Native H2 deployment alpha=0 was used for R, RA, and RCA; RCA CIR was not applied at inference.

The candidate set was frozen from VisA source evidence before Medical access. No target tuning or post-Medical checkpoint replacement is allowed.

| method | epoch | pixel AUROC | pixel AP | image AUROC | image AP |
|---|---:|---:|---:|---:|---:|
| RA | 16 | 0.96042516 | 0.57191153 | 0.83789787 | 0.95796218 |
| RA | 16 | 0.83033206 | 0.49851913 | NA | NA |
| RA | 16 | 0.85749042 | 0.49396902 | NA | NA |
| RA | 16 | 0.79686568 | 0.30762901 | NA | NA |
| RA | 16 | 0.94409254 | 0.04727740 | 0.69025792 | 0.56412618 |
| RA | 16 | 0.88421516 | 0.31044177 | 0.77140197 | 0.71021107 |
| RCA | 12 | 0.96174848 | 0.56083986 | 0.85016209 | 0.96304026 |
| RCA | 12 | 0.80481225 | 0.47089358 | NA | NA |
| RCA | 12 | 0.86752258 | 0.49736167 | NA | NA |
| RCA | 12 | 0.78081279 | 0.31079129 | NA | NA |
| RCA | 12 | 0.93280000 | 0.06599639 | 0.73116701 | 0.58524401 |
| RCA | 12 | 0.86272514 | 0.33540220 | 0.77811999 | 0.76195491 |
| R | 10 | 0.94443874 | 0.40982065 | 0.81992073 | 0.95028191 |
| R | 10 | 0.85546533 | 0.52916957 | NA | NA |
| R | 10 | 0.88407006 | 0.48565991 | NA | NA |
| R | 10 | 0.81388664 | 0.26777876 | NA | NA |
| R | 10 | 0.94524561 | 0.05579908 | 0.62541289 | 0.49112644 |
| R | 10 | 0.92269113 | 0.37832345 | 0.79114612 | 0.79663200 |

## Macro across six Medical targets

| method | epoch | pixel AUROC | pixel AP | image AUROC | image AP |
|---|---:|---:|---:|---:|---:|
| R | 10 | 0.89429959 | 0.35442524 | 0.74549325 | 0.74601345 |
| RA | 16 | 0.87890350 | 0.37162464 | 0.76651925 | 0.74409981 |
| RCA | 12 | 0.86840354 | 0.37354750 | 0.78648303 | 0.77007973 |

MVTec remains NOT RUN until the final architecture/checkpoint decision and one-shot freeze.

Source freeze: `h2_anchor_cir_master_20260901`.
