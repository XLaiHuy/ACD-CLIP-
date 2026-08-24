# P25R2 Execution Contract

Target generation is class-local: native score groups are built once, one image
candidate batch is deployed exactly, and each full candidate image score map is
merged with unchanged native-image groups for exact class AP.  Only compact
target rows persist.  No CLIP or Phase2B training is invoked.

The pre-marker measured route is CUDA batch=16, with target projection 1.118 h
and conservative total projection 1.868 h.  Batch size, scratch layout, and
buffer implementation are engineering choices only; any pre-marker fallback
must satisfy the frozen numerical contract.  After the marker, no code,
backend, feature, target, gate, or policy change is allowed.
