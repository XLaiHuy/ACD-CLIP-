# P19 Exact Parity Contract

P19 accepts only exact P14 semantics. Required checks include AP reference vs
optimized error 0.0, float32 tie parity, direction/harm/threshold/features/
ridge/q/alpha parity, composed SAFE20/E40 parity, one-pass five-q parity, and
global Pearson/Spearman/sign parity. The global stable-sort counterexamples
must produce exactly `+0.8` and `-0.8`, including ties, duplicates, zeros,
negative values, and finite filtering. No quantization, approximation,
sampling, or altered aggregation is permitted.
