# Why the leading future direction could work

The leading direction is representation preservation around the matched Phase2B starting point. The corrected evidence shows a specific asymmetry: C0 is strong on VisA source AP but loses Medical pixel transfer relative to P. An anchor-like constraint could reduce source over-specialization and keep the adapted image/text/prompt geometry nearer to the transfer-capable parent.

The proposal is deliberately minimal. It reuses existing trainable groups and the frozen parent reference, adds no decoder or second backbone, and would be tested source-only with the evaluator unchanged. A successful test would need to preserve source performance while improving source-category-held-out or other transfer proxies; source training improvement alone is insufficient.

This is a hypothesis, not a result. The present commit contains no such loss or training change.
