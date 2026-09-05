# H2 BF16 Seed1 master comparison

`H2_BF16_SCREENING_SEED1_V1` is a one-seed BF16 protocol change. H and A are a matched pair only within this new Seed1 run. The machine-readable table is `H2_BF16_SEED1_MASTER_COMPARISON.csv`; per-dataset/per-class values are in `H2_BF16_SEED1_TARGET_PER_DATASET.csv`.

| Target | Metric | H BF16 Seed1 | A BF16 Seed1 | A − H | Historical Phase2B/H2 best | Published ACD-CLIP |
|---|---|---:|---:|---:|---:|---:|
| Medical | pixel AUROC | 87.1502 | 88.8694 | +1.7192 | 90.98 | UNKNOWN |
| Medical | pixel AP | 30.5611 | 34.6473 | +4.0862 | 40.35 | UNKNOWN |
| Medical | image AUROC | 75.6127 | 75.8583 | +0.2456 | 73.77 | UNKNOWN |
| Medical | image AP | 76.8170 | 77.0199 | +0.2029 | 74.24 | UNKNOWN |
| MVTec AD | pixel AUROC | 84.1344 | 84.8983 | +0.7639 | UNKNOWN | 91.4 |
| MVTec AD | pixel AP | 40.4316 | 41.8373 | +1.4057 | UNKNOWN | 43.6 |
| MVTec AD | image AUROC | 91.1364 | 89.9589 | −1.1775 | UNKNOWN | 90.7 |
| MVTec AD | image AP | 95.7989 | 95.1743 | −0.6246 | UNKNOWN | 95.8 |

H/A values are **MEASURED** and A−H values are **DERIVED**. Historical values are **HISTORICAL_MEASURED**, but they use a different seed and FP16 protocol; Medical Phase2B also used stride-4/rounded evaluation. Published MVTec N=3 values are **PUBLISHED** and only approximately comparable. `PUBLISHED_PROTOCOL_MATCH=PARTIAL`.

A improves both primary pixel metrics over H in this matched BF16 Seed1 screen. The cross-protocol historical/published arithmetic is descriptive only and is not a controlled effect estimate.
