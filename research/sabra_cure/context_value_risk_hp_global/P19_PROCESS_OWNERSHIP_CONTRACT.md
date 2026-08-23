# P19 Process Ownership Contract

`ROLE=parent` may read only marker/progress metadata, JSON compact summaries,
file sizes, and SHA256 digests. It must never deserialize NPZ/tensor
scientific artifacts or own arrays from a child. The parent firewall rejects
scientific-array loads in parent role.

One science child owns one held outer fold. It persists the required immutable
fold artifacts, `fold_summary.json`, and `value_pairs.npz`, validates their
hashes, removes its fold-local temporary cache, then exits. One audit child
independently reconstructs one fold and emits only `fold_audit_summary.json`.
One global worker owns only all `value_pairs.npz` files. One global audit
worker independently repeats that operation. At most one heavy child exists
at a time; parent validates exit and hashes without parsing scientific files.

Parent RSS may not exceed warmed baseline +512 MiB after a child exits. Each
science/audit worker peak is <=14 GiB; global workers <=2 GiB. No completed
child can survive into another child lifecycle.
