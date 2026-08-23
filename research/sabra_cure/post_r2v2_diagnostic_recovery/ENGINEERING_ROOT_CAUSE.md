# P12 Engineering Root Cause

Classification: `PATH_ENVIRONMENT`.

The published P12 terminal commit is an engineering stop with only its attempt
marker and no persisted result artifacts.  Its separate worktree subsequently
contained uncommitted complete-output files although no Python exception,
traceback, or OOM record was persisted.  The P12 execution function writes the
marker, retains all work in memory through its class loop, and only writes
terminal results after that loop.  It has no post-marker exception wrapper,
atomic failure record, durable run log, or progress state.

The exact repository-local failure statement is therefore not a Python source
exception: process supervision returned before a durable terminal state was
observed, while the launched process continued outside that supervising command.
The original direct-file invocation also demonstrated the entrypoint dependency
(`ModuleNotFoundError: No module named 'tools'`) before the marker; the module
entrypoint was then used.  Kernel logs contain no OOM-kill record for the run.

No outcome-bearing P12 result was read for this conclusion.  The repair does
not alter a cohort, formula, feature, alpha, threshold, bin, aggregation, or
decision rule.  It adds only a recovery-local atomic marker/failure/progress/log
lifecycle around the same P12 helper functions.
