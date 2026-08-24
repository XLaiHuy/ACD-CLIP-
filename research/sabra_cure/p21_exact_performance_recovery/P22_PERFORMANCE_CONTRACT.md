# P22 Performance Contract

Exact P20 inventory is `2162` images in frozen class order: candle 200, capsules 160, cashew 150, chewinggum 150, fryum 150, macaroni1 200, macaroni2 200, pcb1 200, pcb2 200, pcb3 201, pcb4 201, pipe_fryum 150.

Worst-case ten-sweep counts are A0 `86480`, A1 `129720`, Stage-D A0 `4324`, and Stage-D A1 `6486` candidates.  The performance ceiling is the maximum route `A0+D`, `A0+A1`, or `A0+A1+D`; it must be <=180 minutes without assuming convergence.  B0--B6 record baseline, CSR/mmap, in-place, preallocation, GPU action batching, two-seed batching, and optional bounded prefetch.  P22 chooses the smallest exact passing variant.

The profile recorded P21 CPU candidate `0.1159484615 s`; exact indexed CUDA batch-4 prototype `0.0120573795 s` per candidate with unique sparse indices and float64 grouped AP.  This supports, but does not replace, the required full exact benchmark.  Class workers remain one; parent queue is bounded to one.  OMP/MKL/OpenBLAS/torch thread settings are frozen in the execution base after benchmark.  No swap growth and bounded RSS/VRAM are required.
