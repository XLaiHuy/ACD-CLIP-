# P21 Coordinate Witness & Exactness Contract

AP is never composed from scalar image deltas. Each action map is constructed once per class, and native-relative sparse score-group count deltas are composed on a one-time exact descending union of float32 scores. Candidate AP uses exact cumulative precision arithmetic. The fast indexed path must agree with the frozen P15 grouped-delta path and direct frozen full-score reference to <=`1e-12`, including final pAUROC, assignments, coverage, wrong-sign, weighted harm, and image ordering.

The pre-marker suite covers all-NATIVE/SAFE20/EXPAND40, mixed states, replacement/revert, ties, identical actions, sparse positive/negative/empty deltas, strict-boundary choice, vectorized candidate parity, and direct group-count parity; applicable cases repeat for SAFE30. No sampling, quantization, approximate AP/sort, GT-driven pruning, global greedy search, random search, or different coordinate ordering is permitted.
