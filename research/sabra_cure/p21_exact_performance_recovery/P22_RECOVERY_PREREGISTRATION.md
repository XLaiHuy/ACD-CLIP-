# P22 Exact P21 Performance Recovery Preregistration

P22 is an engineering-only recovery from P21 terminal `625bf96b192bab713523fbae9b855804b72e5e56` (`P21_PERFORMANCE_NO_GO`).  P21 has no attempt marker and no scientific fold.  P22 preserves P21 actions, coordinates, seeds, strict `> 1e-12` decision rule, LOCO protocol, probes, gates, alpha `.25`, source-only quantiles, image/action orders, and frozen interpretations exactly.

The candidate backend may change only internal storage and execution: class-local exact integer CSR/mmap deltas; in-place CPU apply/evaluate/revert; preallocated scratch; CUDA-resident float64 grouped-count state; batched independent action candidates; and two independent seed lanes.  All final arithmetic remains the P15 grouped-count AP expression.  There is no scalar AP-delta composition, quantization, approximate AP, image reduction, threshold tuning, alpha sweep, or new representation computation.

The selected backend is the smallest exact configuration that passes the contracts: flat CSR/mmap plus CUDA float64 indexed updates with action batching and seed-lane batching.  CUDA is retained only if its real-class and trajectory parity pass and its projected end-to-end runtime is lower than CPU with no memory/swap violation.  CPU remains the exact fallback, not a scientific variant.

Before marker, P22 must pass exactness and trajectory parity against both frozen P15 and P21; use exact 12-class P20 image inventory; project every route with ten sweeps; and meet `max(A0+D, A0+A1, A0+A1+D) <= 180 min`.  Otherwise P22 publishes `P22_PERFORMANCE_NO_GO`, creates no marker, and stops permanently.

Firewall: MVTec 0, Medical 0, additional CLIP forwards 0, Phase2B optimization steps 0, prompt/adapter/LoRA/TTA 0, and no R2-v3/R3/R4.
