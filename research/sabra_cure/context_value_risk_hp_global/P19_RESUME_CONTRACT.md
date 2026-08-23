# P19 Resume Contract

One marker contains UUID, execution SHA, prereg SHA, and frozen input hashes.
Only a mechanical interruption may resume, with exactly the same identity and
only from a verified complete science/audit/global boundary. Completed hashes
must validate and are never recomputed; each resume appends `RESUME_LOG.jsonl`.
A partial science or audit worker is not resumable. Code/protocol/input drift,
an invalid hash, a non-exited child, or any audit failure is
`P19_ENGINEERING_STOP`; no scientific interpretation is permitted.
