# P25R Execution Contract

Targets are generated exactly once (2000/class, 24,000 total) after the
published execution-base marker and reused across LOCO folds. One class-local
target state may retain full score arrays; completed class state is released.
The target engine uses frozen float32 deployment-score semantics and exact AP;
it may optimize only engineering implementation. No per-patch complete
Phase2B/CLIP forward, approximation, or score-order change is allowed.

Before the marker, provenance, panel feasibility, real candle 128-patch
direct-versus-fast parity, critical tests, controller smoke, performance,
memory, firewall, local/remote equality, and a clean worktree must pass.
Performance is preferred <=2 hours; clearly >4 hours is a pre-marker no-go.

The one marker stores UUID, prereg SHA, execution-base SHA, panel hash, and
`runs=1`. Post-marker code/config/feature/target/gate/backend changes or reruns
are forbidden. Progress is atomic and monotonic. Any post-marker infrastructure
failure is terminal and preserves evidence.
