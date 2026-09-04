# H/A confirmatory replication final decision

Status: `ANCHOR_REPLICATION_SUPPORT=NOT_CONFIRMED`.

Seed 1 and seed 2 both completed the frozen H/A E15 training protocol, but
both failed the preregistered hard-validity gate: recorded nonfinite-gradient
skips led to final global steps H=5410 and A=5411. The mismatch is one update
and is directly recorded in each training log.

Because target evaluation is prohibited after a hard-validity failure, there
are no confirmatory Medical or MVTec metrics and no metric-based A-vs-H
replication decision. Seed 0 remains discovery-only evidence and cannot be
used to turn this into a confirmatory claim.

Decision: the family-safe active Anchor improvement is **not confirmed** by
this replication attempt. No redesign, architecture change, or
target-guided rerun is authorized within this run.
