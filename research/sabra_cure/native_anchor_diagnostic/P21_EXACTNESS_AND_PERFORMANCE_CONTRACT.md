# P21 Exactness and Performance Contract

Frozen P15 grouped-count AP is authoritative. GPU may deploy frozen actions or build F1 only if exact score, pAP, and pAUROC parity is zero; CUDA AP is forbidden absent the full preregistered parity contract. Ranker CUDA is used only if >=1.5x benchmark speedup with deterministic prediction/decision parity; otherwise CPU.

Before the marker, benchmark workers {1,2,4}; retain only output-identical configurations with no swap growth and projected aggregate peak RSS <=70% physical RAM. If runtime differs <10%, select fewer workers. P20 thread pins apply if workers>1. Large arrays are class-local; at most one ready class cache exists and completed cache/tensors are released. Preferred total runtime <=90m, acceptable <=2h, hard no-go >3h.

The parent controller is one foreground event-driven process: atomic events only at stage/class/sweep/audit/terminal boundaries, no busy wait or automated status polling.
