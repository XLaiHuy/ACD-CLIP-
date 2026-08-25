# P25R3 Execution Contract

## Immutable target inventory

P25R3 reads these P25R2 target files as frozen inputs:

| Class | SHA-256 |
|---|---|
| candle | `c6322a27c470c374626a51a9beb2f8402fed12e226a475f52d5fbf6131a7e146` |
| capsules | `235003955b3374232859eb7df706e579861e199f57999c89ccf7b76f47f28a39` |
| cashew | `5f67cd2b84ad8b9b5ef0fbcde74de051b6f7107b4f38a6b5100333e316735617` |
| chewinggum | `80bf2ff802297fa5be25dd2175a200c712ffdefa94856c3eb801083bf7c8897d` |
| fryum | `851cbb1a2fa358313e3786842c705eccdef4556a92d59e9544b696ccafb260b4` |
| macaroni1 | `f17ae04814d84dca6b0b907e822853529e7940865f5ab558d81dbd7b9b7b8bff` |
| macaroni2 | `4bc332ecdd61f46114c039a20f372d8c4f19e361ef6099f0aa9ececef7c6ce37` |
| pcb1 | `435caf83b3f09520f0c403966a1a85a6bb9ef2ab7a840b4e99a5510ada83347e` |
| pcb2 | `5b0c2162916b4d04d04227202202473cc66dba32fd0de1f626bdaa421bc4747c` |
| pcb3 | `359b1c2ab204b0bfe99b08faa8759a0e8887843c39e8bc93b57844feff01fbaf` |
| pcb4 | `be64886f65acccfacfc11a35e6177e4cba3916a5c699215550f533a545d0985a` |
| pipe_fryum | `301437a97059d8c823a6fd393e19442ef02b8c0bf6bed341aebcd486beab1ad4` |

Every file must contain exactly 2,000 aligned finite target rows with the
frozen panel fields and order. Historical P25R2 target files remain unchanged.

## Lifecycle

The new runner is isolated from historical P25R2 code. Pre-marker audits cover
objective/gradient parity, ill-conditioned and inactive-column fixtures,
known-failure rejection/recovery, strict JSON, target hashes/schema/alignment,
synthetic Q1/controller/terminal routing, firewall, and runtime/memory.

Publication order is preregistration, execution base, one attempt, terminal
evidence. After the marker there are no code, solver, tolerance, configuration,
worker, or artifact repairs. A failure is preserved as engineering stop.

Runtime is a foreground controller with event-driven progress only. Q1 is
12/12 before aggregation. Q2 is conditional on inherited Q1 gates and, if
entered, uses the same recovered solver uniformly. No continuous polling.
