# Medical comparison report notes

## Reporting job

- Question: report the completed canonical Medical test by dataset and compare both per-dataset values and macro means with published ACD-CLIP.
- Audience: technical research reader.
- Comparison anchor: published ACD-CLIP `ours (N=3)`, because the canonical configuration sets `n_groups=3`.
- Units: all displayed scores are percentages; differences are percentage points.

## Evidence inventory

- Canonical source export: run-relative `canonical_sabra_v1_seed0/final/summary.csv`, completed 2026-08-22 21:34:41 +07:00.
- Canonical provenance: run-relative `canonical_sabra_v1_seed0/final/provenance.json`.
- Published reference: upstream ACD-CLIP README results table, `ours (N=3)` column, mirrored in the repository `README.md` and verified against `https://github.com/upupmake/ACD-CLIP#results` on 2026-08-22.
- Derived snapshot: `comparison.csv` in this directory. Canonical fractional scores were multiplied by 100; published one-decimal percentages were copied as reported; deltas are local minus reference.

## Data-quality checks

- Six of six Medical datasets have completed `metrics.json` and pixel AUROC/AP.
- Brain, Liver, and Retina have defined image AUROC/AP.
- ClinicDB, ColonDB, and Kvasir have only abnormal image labels in this protocol, so image AUROC/AP are undefined and remain blank/N/A rather than zero.
- All six outputs share checkpoint SHA256 `6643cd68eafabf9acdb724242ef5b2d1fbc4bf7e9d2ba7ad6c47776ea646da80` and SABRA freeze SHA256 `3c1ddda3f00751a7a39e8e2d39b87ba8e740f7388a6960255e616e68285471ff`.
- Exact pixel metrics use stride 1. The selected Phase2B checkpoint is epoch 10, seed 0, FP32, effective batch 6.

## Report structure mapping

- Title: first markdown block.
- Technical summary: `technical_summary`.
- Key findings with visual evidence: macro chart, pixel-delta chart, pixel table, image table, each with adjacent interpretation.
- Scope, data, and definitions: `scope_definitions`.
- Methodology: `methodology`.
- Limitations and robustness: `limitations`.
- Recommended next steps: `next_steps`.
- Further questions: `further_questions`.

## Chart map

| Section | Question | Family/type | Fields | Supported claim | Palette policy |
|---|---|---|---|---|---|
| Macro comparison | How do native Phase2B and SABRA compare with published N=3 across four metrics? | Comparison/grouped bar | metric, score_pct, method | SABRA closes much of the image-level gap but not the pixel-AP gap | relaxed multi-category, three series |
| Pixel deltas | Where does SABRA beat or trail the published N=3 reference? | Uncertainty and benchmark/grouped bar around zero | dataset, delta_pp, metric | ColonDB is the only pAP win; Brain and ClinicDB are effectively tied on pAUROC at published precision | hard two-root cap, two metric series |

## Visual QA intent

- Bar charts are used because the evidence is a discrete category comparison, not a time series.
- Scores and deltas are kept in separate charts so absolute levels and signed gaps do not share a misleading scale.
- Exact values remain available in tables and `comparison.csv`.
- Report HTML is generated only through the Data Analytics portable artifact builder.
