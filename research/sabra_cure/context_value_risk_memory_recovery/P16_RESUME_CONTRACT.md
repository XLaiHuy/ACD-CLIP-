# P16 Resume Contract

`ATTEMPT_STARTED.json` holds a new P16 UUID, execution-base SHA, prereg SHA,
and source-input hashes. A mechanical restart is the same attempt only when the
UUID, code SHA, prereg SHA, input hashes, and scientific contract are unchanged;
the latest compact checkpoint validates; completed folds are never recomputed;
and `RESUME_LOG.jsonl` records the restart. Any mismatch refuses resume.

P15 checkpoints are historical engineering evidence only. They cannot seed,
aggregate, or stand in for a P16 scientific fold. SIGTERM preserves only fully
validated compact checkpoints and never marks a partial fold complete.
