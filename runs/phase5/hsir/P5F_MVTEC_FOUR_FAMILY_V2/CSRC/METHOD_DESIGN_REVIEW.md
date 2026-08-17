# CSRC method review

CSRC measures cross-stage rank inconsistency of the same eight B1 peers. It
uses only query-peer distances and fixed peer identities. Spearman uses
average-tie ranks; Kendall is tau-b with explicit tie handling. Pair scope
(all-three or adjacent) and pair aggregation (mean or max) define exactly
eight configs. No peer reselection or stage-specific evidence tuning occurs.
