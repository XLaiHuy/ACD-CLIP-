# P22 Exactness Contract

P22 hashes the immutable scientific references: P21 runner `ad94a70d4f9902d3841770ad5ae30ff9cbc6e74b3410a6756ee4eb710e0d2d9f`, P14 source `d83c59c7e52b21022c90708198048f049bd4e4e46ff443a67adf0c524414273f`, P15 engine `264e384658cb34c9eb2eb4ad1d51e1e9d2c1ce28340b3bac1c206e33c3f0718c`, P20 summary `5b4e6ba6a0be6dbd9aef7826b924ff5d3294f4b0270a4a525a092dbf1d9fae05`, and P21 summary `07cbd00aa27323075458a520bb8491b63013dc0eacd415c54fc2535059b4c8ed`.

Flat delta counts are exact integer score-group deltas.  `index` is `uint32` only after proving the union fits; `positive` and `total` are exact `int32`; conversion to authoritative float64 occurs at the same grouped-count boundary as P21.  Each candidate applies old removal and candidate addition, evaluates grouped AP, and reverts exactly.  Index multiplicity is asserted before indexed CUDA accumulation.

Required pre-marker comparisons to P15 and P21 cover native/safe/expand/mixed states, ties, empty/positive/negative deltas, first/last/multiple images, update/revert, `1e-12` decisions, complete scalar and two-lane trajectories, and a fixed candle non-outcome slice.  Required: AP/pAP error <= `1e-12`, action/trajectory/seed mismatches 0, and exact state restoration.
