# P5B preregistration status

## Status: NOT FROZEN

No P5B candidate is preregistered because Gate 3 failed. Existing B3/B3.1 bridge pairs are GT-defined evaluation pairs and cannot be used as inference constraints. Existing C1 is a full within-cell E reranking and is closed.

## Required next reliability audit (not run)

Before candidate implementation or full evaluation, run a temporary isolated GT-free selector audit over the canonical 2162-image TEST order. Before any GT join, persist `m_bar`, stable pre-action ranks, `D_rank`, `valid_reference`, `E_nonlocal`, peer indices/features, exact score/risk cells, a deterministic proposed-pair trace, accepted/abstained/repeated/conflicting decisions, and native delta/spatial-support accounting. Join GT only after inference/action fields are frozen for descriptive labels.

Command template, not executed:

```text
python /tmp/p5b_gt_free_selector_audit.py --split test --output /tmp/p5b_gt_free_selector_audit_v1
```

If arrays cannot be recovered from existing artifacts, that audit requires one inference-only pass over 2162 images. It is not a candidate evaluation and must not include threshold search.

## Future freeze requirements

A later preregistration may freeze exactly one S/P/G candidate only after it specifies, without GT or test tuning: the exact GT-free pair proposal; relation and exact abstention/tie behavior; disjointness/conflict/repetition handling; exact native delta and unchanged Phase2B quantities; per-patch authority and spatial support; full deployment validation; and the medical-transfer boundary.

No candidate command, threshold, AP gate, or medical evaluation is authorized by this file.
