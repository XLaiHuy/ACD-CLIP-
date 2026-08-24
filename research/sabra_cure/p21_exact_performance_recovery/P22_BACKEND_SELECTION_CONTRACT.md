# P22 Backend Selection Contract

CUDA may be selected only when AP and trajectory parity pass, projected end-to-end runtime is lower than CPU, and no memory/swap failure occurs.  It is preferred when it improves projected wall time by at least 10% or is required for the 180-minute ceiling.  This is execution backend selection only; it does not alter a scientific parameter or result.

Action batching preserves frozen action order and compares each scalar result sequentially with `value > best + 1e-12`.  Two seed lanes are independent SIMD lanes: no state, decision, or outcome is shared.  If any batch result differs, P22 rejects CUDA and records an engineering stop rather than changing arithmetic.
