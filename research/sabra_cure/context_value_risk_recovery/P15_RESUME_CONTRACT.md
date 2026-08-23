# P15 Resume Contract

`ATTEMPT_STARTED.json` stores attempt UUID, execution-base SHA, prereg SHA, and
input hashes. Checkpoints are atomically written and hash-bound. Resume refuses
unless the marker identity, code, preregistration, inputs, and scientific
fields are unchanged; it resumes only from the latest validated checkpoint,
never recomputes a completed checkpoint, and appends `RESUME_LOG.jsonl`.

SIGTERM records a safe checkpoint where one exists and never marks partial work
complete. Any code modification after marker is an engineering stop, not an
invitation to restart a new P15 run.
